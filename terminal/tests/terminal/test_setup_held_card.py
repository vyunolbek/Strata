"""The plan says when the card is held by another program (setup.py: inkling_held_card).

The Inkling plan is made from the card's FREE memory (#6662). A card that is nearly full when `strata launch` runs gets a plan with the
experts in system RAM, which is three times slower, and 0.2.21 said nothing: an RTX 3060 12 GB measured 0.1 GiB free right after a server
had been killed and a launch followed at once, the plan gave the experts 0.00 GiB on the card and 6.76 GiB in RAM, and the engine decoded
at 9.8 tok/s. Now the plan says so, with the numbers, before anything is written."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TERMINAL = Path(__file__).resolve().parents[2]
if str(TERMINAL) not in sys.path:
    sys.path.insert(0, str(TERMINAL))

import setup as S

CARD = {"name": "NVIDIA GeForce RTX 3060", "index": 0, "vram_gb": 12.0, "free_gb": 10.99, "arch": "86"}


class HeldCard(unittest.TestCase):
    def test_a_free_card_is_not_held(self):
        self.assertIsNone(S.inkling_held_card(CARD, S.effective_vram_gb(CARD)))               # 10.99 of 12 free: the desktop's share

    def test_a_nearly_full_card_is_held(self):
        held = S.inkling_held_card({**CARD, "free_gb": 0.1}, 0.1)
        self.assertEqual(held, (0.1, 12.0))
        self.assertEqual(S.inkling_held_card({**CARD, "free_gb": 4.0}, 4.0), (4.0, 12.0))    # a browser holding 8 GiB
        self.assertIsNone(S.inkling_held_card({**CARD, "free_gb": 7.5}, 7.5))                 # 62 % free: not said

    def test_no_free_figure_or_a_unified_card(self):
        self.assertIsNone(S.inkling_held_card({k: v for k, v in CARD.items() if k != "free_gb"}, 12.0))
        self.assertIsNone(S.inkling_held_card({**CARD, "free_gb": 0.0, "uma": True}, 0.0))
        self.assertIsNone(S.inkling_held_card({**CARD, "free_gb": 0.0, "unified": True}, 0.0))
        self.assertIsNone(S.inkling_held_card({**CARD, "free_gb": True}, 1.0))                 # a bool is no figure
        self.assertIsNone(S.inkling_held_card(None, 1.0))
        self.assertIsNone(S.inkling_held_card(CARD, None))                                     # a unified-memory plan names no card

    def test_the_words_carry_the_numbers(self):
        words = S.inkling_held_card_words((0.1, 12.0), {"vram_gb": 0.0, "ram_gb": 6.76, "ssd_gb": 0.0})
        self.assertIn("only 0.1 GiB free of 12.0 GiB right now", words)
        self.assertIn("keeps 0.0 GB of the experts on the card and 6.8 GB in system memory", words)
        self.assertNotIn("SSD", words)
        self.assertIn("strata launch --stop", words)
        words = S.inkling_held_card_words((2.0, 24.0), {"vram_gb": 1.5, "ram_gb": 10.0, "ssd_gb": 3.2})
        self.assertIn("1.5 GB of the experts on the card and 10.0 GB in system memory and 3.2 GB on the SSD", words)


if __name__ == "__main__":
    unittest.main()
