"""The plan sizes the conversation memory (K/V) by the --kv format the person chose, on CUDA.

0.2.21 planned fp8 for every 8-bit format and FP16 when none was named (#7022 did the right thing for HIP only). strata-inkling's k8v4
(INT8 K + rotated 4-bit V) is 12 KiB a token, a quarter less than fp8, so on an RTX 3060 12 GB with SQ-10 at 128K the plan gave the experts
the room the K/V needed: `--kv k8v4 --set kv-offload=off` started with 5.57 GiB of experts and the engine, keeping the K/V on the card,
shrank the context to 65 280 tokens. With the K/V counted (3.37 GiB of experts) the whole 128K fits. `--set kv-offload=off` now reaches
setup's plan too (runner passes --kv-offload), so the plan and the engine agree on where the K/V lives."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

import setup as S
from strata_cli.launch.runner import Runner
from strata_control import inkling_fit as F

CARD = 10.99         # an RTX 3060 12 GB: what is free after the desktop (0.8 GiB more is reserved while it drives a display)
RAM = 30.0


def plan(ctx=131072, build="SQ-10", **kw):
    kw.setdefault("ram_free_gib", 22.0)
    kw.setdefault("desktop", True)
    kw.setdefault("sm", 86)
    return F.place(build, CARD, RAM, False, ctx, True, parallel=1, backend="cuda", **kw)


class BytesPerToken(unittest.TestCase):
    def test_the_global_kv_of_a_format(self):
        self.assertEqual(F.GLOBAL_KV_BYTES_PER_TOK_K8V4, 12288)
        self.assertEqual(F.global_kv_gb(131072, kv_fmt="fp16"), 4.295)
        self.assertEqual(F.global_kv_gb(131072, kv_fmt="fp8"), 2.147)
        self.assertEqual(F.global_kv_gb(131072, kv_fmt="k8v8"), 2.147)     # INT8 K + INT8 V: the fp8 size
        self.assertEqual(F.global_kv_gb(131072, kv_fmt="k8v4"), 1.611)

    def test_no_format_keeps_the_rule(self):
        self.assertEqual(F.global_kv_gb(131072), 4.295)                   # fp16
        self.assertEqual(F.global_kv_gb(131072, fp8_kv=True), 2.147)
        self.assertEqual(F.global_kv_gb(131072, kv_fmt="auto"), 4.295)    # a word that is no format: the rule
        self.assertEqual(F.global_kv_total_gb(131072, kv_fmt="k8v4"), 1.611)
        self.assertEqual(F.kv_cache_gb(131072, fp8_kv=True, kv_fmt="k8v4"), round(1.611 + F.SWA_AND_SCONV_GB_FP8, 3))
        self.assertEqual(F.kv_cache_gb(131072, kv_fmt="k8v4"), round(1.611 + F.SWA_AND_SCONV_GB_FP16, 3))   # the ring follows fp8_kv, not the word

    def test_sycl_keeps_fp16_whatever_is_asked(self):
        self.assertEqual(F.global_kv_gb(131072, fp8_kv=True, backend="sycl"), 4.295)
        self.assertEqual(F.global_kv_gb(131072, kv_fmt="k8v4", backend="sycl"), 4.295)


class CudaPlan(unittest.TestCase):
    def test_k8v4_leaves_more_room_than_the_fp16_the_plan_assumed(self):
        auto = plan()
        self.assertEqual(auto["kv_dtype"], "fp16")
        k8v4 = plan(kv_fmt="k8v4")
        self.assertEqual(k8v4["kv_dtype"], "k8v4")
        self.assertEqual(k8v4["host_kv_gb"], 1.61)
        self.assertEqual(auto["host_kv_gb"], 4.29)
        self.assertEqual(k8v4["vram_gb"], 5.95)                           # the on-device window is what the experts share; the host holds less
        self.assertEqual(plan(kv_fmt="k8v8")["vram_gb"], plan(fp8_kv=True)["vram_gb"])   # k8v8 is fp8
        self.assertEqual(plan(kv_fmt="k8v8")["host_kv_gb"], plan(fp8_kv=True)["host_kv_gb"])

    def test_k8v4_on_the_card_gives_the_experts_what_is_left(self):
        off = plan(kv_fmt="k8v4", kv_offload="off")
        self.assertEqual(off["kv_offload"], "off")
        self.assertEqual(off["host_kv_gb"], 0.0)
        self.assertEqual(off["kv_dtype"], "k8v4")
        self.assertTrue(off["kv_fits"])
        # 10.99 GiB free - 0.30 safety - 0.80 desktop - 1.87 trunk - (1.611 GiB K/V x 1.60 grow + SWA 0.088) - 0.65 DSpark
        self.assertEqual(off["vram_gb"], 3.37)
        fp8 = plan(kv_fmt="fp8", kv_offload="off")
        self.assertEqual(fp8["vram_gb"], 2.51)
        self.assertGreater(off["vram_gb"], fp8["vram_gb"])
        # the old plan: FP16 counted, 5.57 GiB of experts, the K/V sent to host RAM
        self.assertEqual(plan()["vram_gb"], 5.86)

    def test_fp16_on_the_card_does_not_fit_128k_on_a_12_gb_card(self):
        off = plan(kv_fmt="fp16", kv_offload="off")
        self.assertEqual(off["kv_dtype"], "fp16")
        self.assertEqual(off["vram_gb"], 0.0)
        self.assertFalse(off["kv_fits"])                                  # said, not hidden: the engine shrinks the context
        # the same card at 64K with fp16 on the card fits
        small = plan(65536, kv_fmt="fp16", kv_offload="off")
        self.assertTrue(small["kv_fits"])
        self.assertGreater(small["vram_gb"], 0.0)

    def test_the_drafter_is_still_counted(self):
        off = plan(kv_fmt="k8v4", kv_offload="off", dspark=False)
        self.assertGreater(off["vram_gb"], plan(kv_fmt="k8v4", kv_offload="off")["vram_gb"])

    def test_256k_k8v4_on_the_card(self):
        # 262 144 x 12 KiB = 3.22 GiB of K/V, grown by the plan's 1.60 margin: the experts get what is left of the card (the engine measured
        # 2.9 GiB of experts at the full context; the plan keeps more room for the K/V to grow, a slower start that never runs out)
        off = plan(262144, kv_fmt="k8v4", kv_offload="off", dspark=False)
        self.assertTrue(off["kv_fits"])
        self.assertEqual(off["vram_gb"], 1.44)
        self.assertGreater(off["vram_gb"], plan(262144, kv_fmt="fp8", kv_offload="off", dspark=False)["vram_gb"])
        self.assertEqual(plan(262144, kv_fmt="fp16", kv_offload="off")["kv_fits"], False)
        # with the K/V in host RAM the experts keep the window they always had
        self.assertGreater(plan(262144, kv_fmt="k8v4")["vram_gb"], 2.0)


class SetupChoice(unittest.TestCase):
    def kv(self, asked, backend, offload=None):
        return S.inkling_plan_kv(SimpleNamespace(kv=asked, kv_offload=offload), backend)

    def test_cuda_plans_the_format_the_person_chose(self):
        self.assertEqual(self.kv("k8v4", "cuda"), {"fp8_kv": True, "kv_fmt": "k8v4"})
        self.assertEqual(self.kv("k8v8", "cuda"), {"fp8_kv": True, "kv_fmt": "k8v8"})
        self.assertEqual(self.kv("fp8", "cuda"), {"fp8_kv": True, "kv_fmt": "fp8"})
        self.assertEqual(self.kv("fp16", "cuda"), {"fp8_kv": False, "kv_fmt": "fp16"})
        self.assertEqual(self.kv("", "cuda"), {})
        self.assertEqual(self.kv(None, "cuda"), {})
        self.assertEqual(self.kv("auto", "cuda"), {})

    def test_kv_offload_reaches_the_plan_on_cuda(self):
        self.assertEqual(self.kv("k8v4", "cuda", "off"), {"fp8_kv": True, "kv_fmt": "k8v4", "kv_offload": "off"})
        self.assertEqual(self.kv(None, "cuda", "off"), {"kv_offload": "off"})
        self.assertEqual(self.kv(None, "cuda", "on"), {"kv_offload": "on"})
        self.assertEqual(self.kv(None, "cuda", "auto"), {})             # the engine's own rule
        self.assertEqual(self.kv(None, "cuda", ""), {})
        self.assertEqual(self.kv("fp8", "hip", "off"), {"fp8_kv": True})  # HIP keeps its K/V in the card by rule: nothing to say

    def test_metal_sycl_and_an_unknown_backend_are_left_alone(self):
        for backend in ("sycl", "metal", None):
            for asked in ("fp8", "fp16", "k8v8", "k8v4", ""):
                self.assertEqual(self.kv(asked, backend), {}, (backend, asked))


class RunnerPassesIt(unittest.TestCase):
    def args_for(self, extra):
        from strata_cli.launch import runner as R
        from strata_cli.launch import settings as ST
        from tests.terminal.launch_support import HW_12, Home, catalog

        with Home():
            s = ST.LaunchSettings(family="inkling-2-mini", model="SQ-10", context=131072, kv="k8v4", vision="off", extra=extra)
            return Runner(s, catalog=catalog(HW_12)).setup_args()

    def test_setup_args_carry_kv_offload(self):
        args = self.args_for({"--kv-offload": "off"})
        self.assertEqual(args[args.index("--kv") + 1], "k8v4")
        self.assertEqual(args[args.index("--kv-offload") + 1], "off")
        self.assertNotIn("--kv-offload", self.args_for({}))
        self.assertNotIn("--kv-offload", self.args_for({"--kv-offload": "auto"}))
        self.assertEqual(self.args_for({"--kv-offload": "on"})[self.args_for({"--kv-offload": "on"}).index("--kv-offload") + 1], "on")


if __name__ == "__main__":
    unittest.main()
