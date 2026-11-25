"""Efficiency tests: running time grows as O(n * m), memory stays bounded.

The time limits are generous (several times the time measured on a laptop)
so that the tests stay reliable on slow CI machines; they still catch an
accidental change of complexity, e.g. an O(n * m) step done once per cell.
Run from the project folder with:  python -m unittest -v
"""

from __future__ import annotations

import random
import sys
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smith_waterman import core, get_matrix, identity_matrix, smith_waterman
from smith_waterman.core import Alignment
from smith_waterman.formatting import format_alignment

PROTEIN_LETTERS = "ARNDCQEGHILKMFPSTWYV"


def random_sequence(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def best_time(function, *args, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function(*args)
        best = min(best, time.perf_counter() - start)
    return best


class Scaling(unittest.TestCase):
    """Doubling both lengths must multiply the work by about four."""

    def check_quadratic(self, alphabet, matrix, d, e):
        rng = random.Random(7)
        small = [random_sequence(rng, alphabet, 200) for _ in range(2)]
        large = [random_sequence(rng, alphabet, 400) for _ in range(2)]
        t_small = best_time(smith_waterman, *small, matrix, d, e)
        t_large = best_time(smith_waterman, *large, matrix, d, e)
        ratio = t_large / t_small
        # 4 in theory; anything below 8 rules out an O(n^2 m) or O(n m^2) slip.
        self.assertLess(ratio, 8, f"x{ratio:.1f} when doubling the lengths")
        self.assertGreater(ratio, 2, f"x{ratio:.1f}: the timing is not measuring the DP fill")

    def test_linear_dna_is_quadratic(self):
        self.check_quadratic("ACGT", identity_matrix(1, -1), 2, None)

    def test_affine_protein_is_quadratic(self):
        self.check_quadratic(PROTEIN_LETTERS, get_matrix("BLOSUM62"), 10, 1)


class TimeBudget(unittest.TestCase):
    """Real-size inputs finish well within an interactive delay."""

    def test_500_x_500_protein(self):
        rng = random.Random(8)
        a, b = (random_sequence(rng, PROTEIN_LETTERS, 500) for _ in range(2))
        matrix = get_matrix("BLOSUM62")
        # About 0.07 s (linear) and 0.15 s (affine) on a laptop.
        self.assertLess(best_time(smith_waterman, a, b, matrix, 8, None, repeats=1), 1.5)
        self.assertLess(best_time(smith_waterman, a, b, matrix, 10, 1, repeats=1), 3.0)

    def test_traceback_cost_does_not_depend_on_the_matrix_size(self):
        """Many optimal alignments are rebuilt quickly: the traceback walks only its own path."""
        a = "ACGT" * 150                      # 600 symbols, the best score is reached many times
        result = smith_waterman(a, "ACGT", identity_matrix(1, -1), 2)
        self.assertEqual(len(result.alignments), 150)
        start = time.perf_counter()
        for alignment in result.alignments:
            core.traceback_linear(result.a, result.b, result.matrix, result.S, 2,
                                  (alignment.end_a, alignment.end_b))
        self.assertLess(time.perf_counter() - start, 0.5)


class Memory(unittest.TestCase):
    def peak_bytes(self, *args) -> int:
        tracemalloc.start()
        try:
            smith_waterman(*args)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_peak_memory_per_cell(self):
        rng = random.Random(9)
        a, b = (random_sequence(rng, PROTEIN_LETTERS, 300) for _ in range(2))
        matrix = get_matrix("BLOSUM62")
        cells = 301 * 301
        # Measured: about 9 bytes per cell (linear, one matrix) and 51 (affine, three).
        self.assertLess(self.peak_bytes(a, b, matrix, 8, None) / cells, 25)
        self.assertLess(self.peak_bytes(a, b, matrix, 10, 1) / cells, 110)

    def test_no_traceback_matrix_is_kept(self):
        """Only the score matrices are stored; the traceback recomputes its moves."""
        S = core.fill_linear("ACGT", "AGT", identity_matrix(), 1)
        self.assertEqual(len(S), 5)
        self.assertTrue(all(isinstance(row, list) and len(row) == 4 for row in S))
        self.assertEqual(len(core.fill_affine("ACGT", "AGT", identity_matrix(), 2, 1)), 3)

    def test_substitution_rows_are_shared_between_equal_symbols(self):
        rows = core._substitution_rows("ACAACA", "GATT", identity_matrix())
        self.assertIs(rows[0], rows[2])
        self.assertIs(rows[1], rows[4])
        self.assertEqual(len({id(row) for row in rows}), 2)


class Formatting(unittest.TestCase):
    def test_midline_is_built_once_per_alignment(self):
        """format_alignment must not rebuild the O(L) midline for every block."""
        a = "ACGT" * 100
        alignment = smith_waterman(a, a, identity_matrix(), 2).alignment
        with mock.patch.object(Alignment, "midline", new_callable=mock.PropertyMock,
                               return_value=alignment.midline) as midline:
            format_alignment(alignment, width=10)
        self.assertEqual(midline.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
