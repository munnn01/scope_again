"""Verify exact alignment with the ACCV 2026 SCOPE paper and released checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from scope.codecs import codec_proxy_single
from scope.data import CocoBoxCropDataset
from scope.evaluation.conditioning import infer_severity_from_experiment
from scope.inference import apply_scope_bgr, load_scope_model
from scope.model import SCOPE, normalize_conditioning_mode
from scope.train_scope import build_argparser
from scope.training import compute_losses, print_and_assert_critical_config


ROOT = Path(__file__).resolve().parents[1]


class PaperArchitectureAndParameterTests(unittest.TestCase):
    def test_default_scope_matches_paper_architecture_and_param_count(self):
        model = SCOPE(num_codec_families=3)
        self.assertEqual(model.conditioning_mode, "family_severity")
        self.assertEqual(model.bg_mode, "blur_replace")
        self.assertEqual(model.bg_blur_kernel, 51)
        self.assertEqual(model.bg_lowres_scale, 16)
        self.assertAlmostEqual(model.max_delta_obj, 0.09)
        self.assertAlmostEqual(model.max_delta_bg, 0.03)
        self.assertAlmostEqual(float(model.attn_head.bias.detach().item()), -4.0, places=5)

        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 1_879_119)

    def test_table3_conditioning_ablation_parameter_counts(self):
        expected = [
            ("family_severity", 3, 1_879_119),
            ("codec_family_only", 3, 1_878_735),
            ("severity_only", 3, 1_878_583),
            ("none", 1, 1_742_935),
        ]
        for mode, num_fams, param_count in expected:
            with self.subTest(mode=mode):
                model = SCOPE(num_codec_families=num_fams, conditioning_mode=mode)
                actual = sum(p.numel() for p in model.parameters())
                self.assertEqual(actual, param_count)

    def test_severity_basis_matches_paper_equations_1_and_2(self):
        sev = torch.tensor([-0.25, 0.0, 0.50, 0.625, 0.75, 0.90, 0.95, 1.0, 1.5])
        basis = SCOPE._expand_scalar_severity(sev)
        self.assertEqual(tuple(basis.shape), (9, 6))

        # Clamped < 0 -> all zeros; clamped > 1 -> all ones
        self.assertTrue(torch.allclose(basis[0], torch.zeros(6)))
        self.assertTrue(torch.allclose(basis[-1], torch.ones(6)))

        # At s = 0.625: hi50 = (0.625 - 0.50)/0.50 = 0.25, hi75 = 0, hi90 = 0
        expected_0625 = torch.tensor([0.625, 0.625**2, 0.625**3, 0.25, 0.0, 0.0])
        self.assertTrue(torch.allclose(basis[3], expected_0625, atol=1e-6))

        # At s = 0.95: hi50 = 0.90, hi75 = 0.80, hi90 = 0.50
        expected_095 = torch.tensor([0.95, 0.95**2, 0.95**3, 0.90, 0.80, 0.50])
        self.assertTrue(torch.allclose(basis[6], expected_095, atol=1e-6))

    def test_odd_and_tiny_resolutions_and_residual_bounds(self):
        model = SCOPE(num_codec_families=3, base_ch=8).eval()
        for height, width in ((16, 16), (37, 53)):
            with self.subTest(shape=(height, width)):
                x = torch.rand(2, 3, height, width)
                sev = torch.tensor([0.0, 1.0])
                fam = torch.tensor([0, 2])
                with torch.no_grad():
                    edited, delta_obj, delta_bg, attn, eff_delta = model(x, sev, fam)
                self.assertEqual(tuple(edited.shape), (2, 3, height, width))
                self.assertEqual(tuple(attn.shape), (2, 1, height, width))
                self.assertTrue(torch.isfinite(edited).all())
                self.assertGreaterEqual(float(edited.min()), 0.0)
                self.assertLessEqual(float(edited.max()), 1.0)
                self.assertLessEqual(float(delta_obj.abs().max()), 0.09 + 1e-6)
                self.assertGreaterEqual(float(attn.min()), 0.0)
                self.assertLessEqual(float(attn.max()), 1.0)
                self.assertTrue(torch.allclose(eff_delta, edited - x, atol=1e-6))
                self.assertIsNotNone(delta_bg)


class PaperHyperparameterAlignmentTests(unittest.TestCase):
    def test_cli_defaults_match_paper_and_train_shell(self):
        args = build_argparser().parse_args([
            "--images-dir", "train_imgs",
            "--annotations-json", "train.json",
            "--output-dir", "runs/test",
            "--assert-objstrong",
        ])
        self.assertEqual(args.crop_size, 512)
        self.assertAlmostEqual(args.object_crop_prob, 0.60)
        self.assertAlmostEqual(args.crop_margin, 4.0)
        self.assertEqual(args.codec_families, "diff_jpeg,bmshj,bpg_like")
        self.assertEqual(args.severity_levels, "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
        self.assertEqual(args.val_severity_levels, "0.0,0.2,0.4,0.6,0.8,1.0")
        self.assertEqual(args.diffjpeg_quality_min, 1)
        self.assertEqual(args.diffjpeg_quality_max, 99)
        self.assertEqual(args.bmshj_model, "bmshj2018-hyperprior")
        self.assertEqual(args.bmshj_quality_min, 1)
        self.assertEqual(args.bmshj_quality_max, 8)
        self.assertAlmostEqual(args.bmshj_bpp_scale, 0.03)
        self.assertAlmostEqual(args.bpg_qp_min, 0.0)
        self.assertAlmostEqual(args.bpg_qp_max, 51.0)
        self.assertEqual(args.bpg_proxy_mode, "medium")

        self.assertEqual(args.base_ch, 48)
        self.assertEqual(args.family_emb_dim, 8)
        self.assertEqual(args.cond_dim, 64)
        self.assertAlmostEqual(args.max_delta_obj, 0.09)
        self.assertAlmostEqual(args.max_delta_bg, 0.03)
        self.assertAlmostEqual(args.attention_bias, -4.0)
        self.assertEqual(args.bg_mode, "blur_replace")
        self.assertEqual(args.bg_blur_kernel, 51)
        self.assertEqual(args.conditioning_mode, "family_severity")

        self.assertEqual(args.epochs, 40)
        self.assertEqual(args.batch, 8)
        self.assertEqual(args.samples_per_epoch, 12000)
        self.assertAlmostEqual(args.lr, 1e-4)
        self.assertAlmostEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.scheduler, "cosine")
        self.assertEqual(args.warmup_epochs, 3)
        self.assertAlmostEqual(args.grad_clip, 1.0)

        self.assertEqual(args.attn_blur, 3)
        self.assertEqual(args.det_mask_blur, 41)
        self.assertAlmostEqual(args.lambda_bg_flat, 0.25)
        self.assertAlmostEqual(args.lambda_bg_smooth, 0.35)
        self.assertAlmostEqual(args.lambda_rate, 0.50)
        self.assertAlmostEqual(args.lambda_attn, 1.00)
        self.assertAlmostEqual(args.lambda_attn_sparse, 0.15)
        self.assertAlmostEqual(args.lambda_attn_area, 0.50)
        self.assertAlmostEqual(args.lambda_delta, 0.02)
        self.assertAlmostEqual(args.lambda_delta_tv, 0.05)
        self.assertAlmostEqual(args.lambda_det, 4.0)
        self.assertEqual(args.detector_backend, "detectron2")
        self.assertEqual(args.detectron2_config, "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.assertEqual(args.detectron2_feature_names, "p2,p3,p4,p5,p6")
        self.assertAlmostEqual(args.d2_fpn_mask_weight, 0.70)
        self.assertAlmostEqual(args.d2_fpn_full_weight, 0.30)

        # Must pass critical paper config assertion without raising
        print_and_assert_critical_config(args)

    def test_dataset_init_defaults_match_paper(self):
        import inspect
        sig = inspect.signature(CocoBoxCropDataset.__init__)
        self.assertEqual(sig.parameters["crop_size"].default, 512)
        self.assertAlmostEqual(sig.parameters["object_crop_prob"].default, 0.60)
        self.assertAlmostEqual(sig.parameters["crop_margin"].default, 4.0)

    def test_inferred_severities_match_all_yaml_operating_points(self):
        for cfg_name in ("eval.yaml", "ablation.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / cfg_name).read_text(encoding="utf-8"))
            for exp in cfg["experiments"]:
                if not exp["mode"].startswith("scope_"):
                    continue
                explicit = float(exp["condition"]["severity"])
                stripped = {k: v for k, v in exp.items() if k not in ("condition", "severity")}
                inferred = infer_severity_from_experiment(stripped)
                with self.subTest(config=cfg_name, exp=exp["name"]):
                    self.assertAlmostEqual(inferred, explicit, places=5)

    def test_handmade_codec_proxies_support_boundary_severities(self):
        x = torch.rand(1, 3, 32, 32)
        for family in ("diff_jpeg", "bpg_like", "bmshj", "jp2_like"):
            for sev in (0.0, 0.5, 1.0):
                with self.subTest(family=family, severity=sev):
                    y = codec_proxy_single(x, family_name=family, severity=torch.tensor(sev))
                    self.assertEqual(tuple(y.shape), (1, 3, 32, 32))
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertGreaterEqual(float(y.min()), 0.0)
                    self.assertLessEqual(float(y.max()), 1.0)


class ConditioningModesAndEdgeCaseTests(unittest.TestCase):
    def test_all_four_conditioning_modes_checkpoint_roundtrip_and_inference(self):
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        modes = ("family_severity", "codec_family_only", "severity_only", "none")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for include_meta in (True, False):
                for mode in modes:
                    with self.subTest(mode=mode, include_meta=include_meta):
                        model = SCOPE(num_codec_families=3, base_ch=8, conditioning_mode=mode).eval()
                        ckpt_path = root / f"{mode}_{int(include_meta)}.pt"
                        payload = {
                            "model": model.state_dict(),
                            "codec_family_to_id": {"diff_jpeg": 0, "bmshj": 1, "bpg_like": 2},
                            "base_ch": 8,
                        }
                        if include_meta:
                            payload["conditioning_mode"] = mode
                        torch.save(payload, ckpt_path)

                        loaded, _ = load_scope_model(str(ckpt_path), torch.device("cpu"))
                        self.assertEqual(loaded.conditioning_mode, mode)

                        out = apply_scope_bgr(image, str(ckpt_path), "diff_jpeg", 0.5, "cpu")
                        self.assertEqual(out.shape, image.shape)
                        self.assertEqual(out.dtype, np.uint8)

    def test_all_conditioning_modes_backprop_cleanly(self):
        args = build_argparser().parse_args([
            "--images-dir", "x", "--annotations-json", "y", "--output-dir", "z",
            "--device", "cpu", "--lambda-det", "0",
        ])
        for mode in ("family_severity", "codec_family_only", "severity_only", "none"):
            with self.subTest(mode=mode):
                model = SCOPE(num_codec_families=3, base_ch=8, conditioning_mode=mode)
                images = torch.rand(2, 3, 32, 32)
                masks = torch.zeros(2, 1, 32, 32)
                masks[:, :, 8:24, 8:24] = 1.0
                loss, _ = compute_losses(
                    args=args,
                    model=model,
                    codec_proxy_manager=None,
                    detector_loss_module=None,
                    images=images,
                    masks=masks,
                    family_ids=torch.tensor([0, 2]),
                    severity_norm=torch.tensor([0.2, 0.8]),
                    id_to_codec_family={0: "diff_jpeg", 1: "bmshj", 2: "bpg_like"},
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertGreater(len(grads), 0)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_error_paths_and_validation(self):
        with self.assertRaises(ValueError):
            normalize_conditioning_mode("invalid_mode")
        with self.assertRaises(ValueError):
            SCOPE(num_codec_families=3, bg_mode="not_a_mode")

        model_full = SCOPE(num_codec_families=2, base_ch=8, conditioning_mode="family_severity")
        x = torch.rand(1, 3, 16, 16)
        with self.assertRaises(ValueError):
            model_full(x, severity=None, codec_family=torch.tensor([0]))
        with self.assertRaises(ValueError):
            model_full(x, severity=torch.tensor([0.5]), codec_family=None)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m.pt"
            torch.save({
                "model": model_full.state_dict(),
                "codec_family_to_id": {"diff_jpeg": 0, "bmshj": 1},
                "base_ch": 8,
            }, ckpt)
            img = np.zeros((16, 16, 3), dtype=np.uint8)
            with self.assertRaises(ValueError):
                apply_scope_bgr(img, str(ckpt), "diff_jpeg", -0.1, "cpu")
            with self.assertRaises(ValueError):
                apply_scope_bgr(img, str(ckpt), "diff_jpeg", 1.1, "cpu")
            with self.assertRaises(ValueError):
                apply_scope_bgr(img, str(ckpt), "unknown_codec", 0.5, "cpu")


if __name__ == "__main__":
    unittest.main()
