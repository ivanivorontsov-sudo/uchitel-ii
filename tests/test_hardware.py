import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from hardware import need_gb, pick, recommend


class HardwareRecommendationTests(unittest.TestCase):
    def test_multi_gpu_does_not_sum_vram_for_ddp(self):
        # DDP replicates the full model on each card. 2x12 GB must not
        # be treated as one 24 GB device.
        cards = [
            {"vram_gb": 12.0, "free_gb": 10.0},
            {"vram_gb": 12.0, "free_gb": 9.0},
        ]
        result = recommend(cards, 16.0)
        qwen7b = next(x for x in result["models"] if x["id"] == "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(qwen7b["method"], "qlora")
        self.assertEqual(result["vram_total_gb"], 24.0)
        self.assertEqual(result["vram_per_gpu_free_gb"], 9.0)

    def test_uses_free_not_total_vram(self):
        self.assertEqual(pick(3.0, vram=4.0, ram=32.0), "qlora")

    def test_cpu_recommendation_uses_available_ram(self):
        self.assertIsNone(pick(0.5, vram=0.0, ram=3.0))
        self.assertEqual(pick(0.5, vram=0.0, ram=8.0), "cpu")

    def test_memory_estimates_are_positive(self):
        for method in ("full", "lora", "qlora", "cpu"):
            self.assertGreater(need_gb(0.5, method), 0)


if __name__ == "__main__":
    unittest.main()
