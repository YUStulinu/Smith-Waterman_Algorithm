"""Unit tests. Run from the project folder with:  python -m unittest -v"""

from __future__ import annotations

import contextlib
import io
import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smith_waterman import (  # noqa: E402
    NUCLEOTIDE,
    PROTEIN,
    clean_sequence,
    detect_type,
    get_matrix,
    identity_matrix,
    prepare_sequence,
    read_fasta,
    score_alignment,
    smith_waterman,
    transition_transversion_matrix,
)
from smith_waterman.cli import main as cli_main  # noqa: E402
from smith_waterman.core import fill_affine, fill_linear, find_best_cells  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NEG_INF = float("-inf")


def reference_linear(a, b, matrix, d):
    """S for linear gaps, transcribed directly from the recurrence (full matrix, max())."""
    S = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            # max() keeps the first of equal values: diagonal, up, left, as in the course
            best = max(S[i - 1][j - 1] + matrix.score(a[i - 1], b[j - 1]), S[i - 1][j] - d, S[i][j - 1] - d)
            S[i][j] = best if best > 0 else 0
    return S


def reference_affine(a, b, matrix, d, e):
    """S, Ga, Gb for affine gaps, transcribed directly from the three recurrences."""
    n, m = len(a), len(b)
    S = [[0] * (m + 1) for _ in range(n + 1)]
    Ga = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Gb = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            Gb[i][j] = max(S[i - 1][j] - d, Gb[i - 1][j] - e, Ga[i - 1][j] - d)
            Ga[i][j] = max(S[i][j - 1] - d, Ga[i][j - 1] - e, Gb[i][j - 1] - d)
            best = max(S[i - 1][j - 1], Gb[i - 1][j - 1], Ga[i - 1][j - 1]) + matrix.score(a[i - 1], b[j - 1])
            S[i][j] = best if best > 0 else 0
    return S, Ga, Gb


def brute_force_best_score(a, b, matrix, d, e=None):
    """Best local score by exhaustive search over all substring pairs.

    Every pair of substrings is aligned globally with a plain recursive
    enumeration of all alignments; only usable for very short sequences.
    """
    best = 0
    for i0 in range(len(a)):
        for i1 in range(i0 + 1, len(a) + 1):
            for j0 in range(len(b)):
                for j1 in range(j0 + 1, len(b) + 1):
                    for x, y in _all_alignments(a[i0:i1], b[j0:j1]):
                        best = max(best, score_alignment(x, y, matrix, d, e))
    return best


def _all_alignments(a, b):
    if not a and not b:
        yield "", ""
        return
    if a and b:
        for x, y in _all_alignments(a[1:], b[1:]):
            yield a[0] + x, b[0] + y
    if a:
        for x, y in _all_alignments(a[1:], b):
            yield a[0] + x, "-" + y
    if b:
        for x, y in _all_alignments(a, b[1:]):
            yield "-" + x, b[0] + y


class KnownExamples(unittest.TestCase):
    def test_course_example(self):
        """Course 4: a = ATCTGA, b = TGCA, match 1, mismatch -1, gap 2 -> score 2."""
        result = smith_waterman("ATCTGA", "TGCA", identity_matrix(1, -1), 2)
        self.assertEqual(result.best_score, 2)
        self.assertEqual(result.S, [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 1, 1],
        ])
        alignment = result.alignment
        self.assertEqual((alignment.aligned_a, alignment.aligned_b), ("TG", "TG"))
        self.assertEqual((alignment.start_a, alignment.end_a), (4, 5))
        self.assertEqual((alignment.start_b, alignment.end_b), (1, 2))

    def test_lab_example(self):
        """Lab 3-4 / Wikipedia example: TGTTACGG vs GGTTGACTA, +3 / -3, gap 2."""
        result = smith_waterman("TGTTACGG", "GGTTGACTA", identity_matrix(3, -3), 2)
        self.assertEqual(result.best_score, 13)
        self.assertEqual((result.alignment.aligned_a, result.alignment.aligned_b), ("GTT-AC", "GTTGAC"))

    def test_durbin_example(self):
        """Durbin et al. (1998), fig. 2.6: BLOSUM50, gap 8 -> AWGHE / AW-HE, score 28."""
        result = smith_waterman("HEAGAWGHEE", "PAWHEAE", get_matrix("BLOSUM50"), 8)
        self.assertEqual(result.best_score, 28)
        self.assertEqual((result.alignment.aligned_a, result.alignment.aligned_b), ("AWGHE", "AW-HE"))

    def test_identical_sequences(self):
        sequence = "ACGTACGTTAGC"
        result = smith_waterman(sequence, sequence, identity_matrix(2, -1), 3)
        self.assertEqual(result.best_score, 2 * len(sequence))
        self.assertEqual(result.alignment.aligned_a, sequence)
        self.assertEqual(result.alignment.gaps, 0)

    def test_no_similarity(self):
        result = smith_waterman("AAAA", "TTTT", identity_matrix(1, -1), 2)
        self.assertEqual(result.best_score, 0)
        self.assertIsNone(result.alignment)

    def test_all_optimal_alignments_reported(self):
        result = smith_waterman("ACGTTACG", "ACG", identity_matrix(1, -1), 2)
        self.assertEqual(result.best_score, 3)
        self.assertEqual(len(result.alignments), 2)
        self.assertEqual([x.start_a for x in result.alignments], [1, 6])


class AffineGaps(unittest.TestCase):
    def test_affine_bridges_a_long_gap(self):
        """A 3-symbol deletion: affine gaps keep the two halves together."""
        matrix = identity_matrix(1, -1)
        a, b = "ACGTTTGACCATGCA", "ACGTTTCATGCA"

        linear = smith_waterman(a, b, matrix, 3)
        self.assertEqual(linear.best_score, 6)
        self.assertEqual(linear.alignment.gaps, 0)

        affine = smith_waterman(a, b, matrix, 3, 1)
        self.assertEqual(affine.best_score, 12 - (3 + 1 + 1))
        self.assertEqual(affine.alignment.aligned_a, "ACGTTTGACCATGCA")
        self.assertEqual(affine.alignment.aligned_b, "ACGTTT---CATGCA")

    def test_affine_with_equal_penalties_is_linear(self):
        rng = random.Random(1)
        matrix = identity_matrix(2, -1)
        for _ in range(200):
            a = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 30)))
            b = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 30)))
            d = rng.choice([1, 2, 3])
            self.assertEqual(
                smith_waterman(a, b, matrix, d).best_score,
                smith_waterman(a, b, matrix, d, d).best_score,
            )

    def test_cheaper_extension_never_lowers_the_score(self):
        rng = random.Random(2)
        matrix = get_matrix("BLOSUM62")
        letters = "ARNDCQEGHILKMFPSTWYV"
        for _ in range(100):
            a = "".join(rng.choice(letters) for _ in range(rng.randint(5, 40)))
            b = "".join(rng.choice(letters) for _ in range(rng.randint(5, 40)))
            self.assertGreaterEqual(
                smith_waterman(a, b, matrix, 10, 1).best_score,
                smith_waterman(a, b, matrix, 10).best_score,
            )

    def test_extension_larger_than_opening_is_rejected(self):
        with self.assertRaises(ValueError):
            smith_waterman("ACGT", "ACGT", identity_matrix(), 1, 2)


class Correctness(unittest.TestCase):
    """Properties that must hold for every input."""

    def check_result(self, result, d, e):
        for alignment in result.alignments:
            # 1. the reported score is the real score of the reported alignment
            rescored = score_alignment(alignment.aligned_a, alignment.aligned_b, result.matrix, d, e)
            self.assertAlmostEqual(rescored, result.best_score)
            # 2. removing the gaps gives back the reported substrings
            self.assertEqual(alignment.aligned_a.replace("-", ""), result.a[alignment.start_a - 1:alignment.end_a])
            self.assertEqual(alignment.aligned_b.replace("-", ""), result.b[alignment.start_b - 1:alignment.end_b])
            # 3. a local alignment never starts or ends with a gap
            for text in (alignment.aligned_a, alignment.aligned_b):
                self.assertNotEqual(text[0], "-")
                self.assertNotEqual(text[-1], "-")
        # 4. no cell of S is negative
        self.assertGreaterEqual(min(min(row) for row in result.S), 0)

    def test_random_sequences(self):
        rng = random.Random(3)
        matrices = [identity_matrix(1, -1), identity_matrix(5, -4), transition_transversion_matrix()]
        for _ in range(300):
            a = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 40)))
            b = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 40)))
            matrix = rng.choice(matrices)
            d = rng.choice([1, 2, 4, 6])
            e = rng.choice([None, 0.5, 1])
            self.check_result(smith_waterman(a, b, matrix, d, e), d, e)

    def test_symmetry(self):
        rng = random.Random(4)
        matrix = get_matrix("BLOSUM62")
        letters = "ARNDCQEGHILKMFPSTWYV"
        for _ in range(100):
            a = "".join(rng.choice(letters) for _ in range(rng.randint(1, 30)))
            b = "".join(rng.choice(letters) for _ in range(rng.randint(1, 30)))
            for e in (None, 1):
                self.assertEqual(
                    smith_waterman(a, b, matrix, 8, e).best_score,
                    smith_waterman(b, a, matrix, 8, e).best_score,
                )

    def test_against_exhaustive_search(self):
        """The DP score equals the best score over ALL local alignments."""
        rng = random.Random(5)
        matrix = identity_matrix(2, -1)
        for _ in range(40):
            a = "".join(rng.choice("ACG") for _ in range(rng.randint(1, 5)))
            b = "".join(rng.choice("ACG") for _ in range(rng.randint(1, 5)))
            for d, e in itertools.product([1, 3], [None, 1]):
                expected = brute_force_best_score(a, b, matrix, d, e)
                self.assertEqual(smith_waterman(a, b, matrix, d, e).best_score, expected, (a, b, d, e))


class ReferenceRecurrences(unittest.TestCase):
    """The optimised fill loops give, cell by cell, what the recurrences give."""

    def assert_same_grid(self, got, expected, context):
        self.assertEqual(len(got), len(expected), context)
        for i, (row, expected_row) in enumerate(zip(got, expected)):
            # same values and same int/float type in every cell
            self.assertEqual([(type(v), v) for v in row], [(type(v), v) for v in expected_row], (context, i))

    def random_cases(self, seed, count):
        rng = random.Random(seed)
        matrices = [identity_matrix(1, -1), identity_matrix(2, -1), identity_matrix(1.5, -0.25),
                    identity_matrix(0.3, -0.1), transition_transversion_matrix(), get_matrix("NUC.4.4")]
        protein = get_matrix("BLOSUM62")
        for _ in range(count):
            if rng.random() < 0.3:
                matrix, letters = protein, "ARNDCQEGHILKMFPSTWYV"
            else:
                matrix, letters = rng.choice(matrices), "ACGT"
            a = "".join(rng.choice(letters) for _ in range(rng.randint(1, 25)))
            b = "".join(rng.choice(letters) for _ in range(rng.randint(1, 25)))
            d = rng.choice([0, 0.1, 0.5, 1, 2, 2.5, 4, 8])
            e = rng.choice([0, 0.1, 0.5, 1, d])
            yield a, b, matrix, d, min(d, e)

    def test_linear(self):
        for a, b, matrix, d, _e in self.random_cases(6, 300):
            self.assert_same_grid(fill_linear(a, b, matrix, d), reference_linear(a, b, matrix, d), (a, b, d))

    def test_affine(self):
        for a, b, matrix, d, e in self.random_cases(7, 300):
            for got, expected, name in zip(fill_affine(a, b, matrix, d, e), reference_affine(a, b, matrix, d, e),
                                           ("S", "Ga", "Gb")):
                self.assert_same_grid(got, expected, (name, a, b, d, e))

    def test_rounding_tie(self):
        """Ga(i-1, j) = 2.8e-17 > S(i-1, j) = 0, yet both give -4 after subtracting d = 4."""
        a, b = "TTTT", "TGTGTGGGGGGTTGGGTGGTTTGTTGGTGTGTGTTGGTGGGGGGGGTTGGTTGTTTGTGGGGTG"
        matrix = identity_matrix(1.5, -0.25)
        for got, expected in zip(fill_affine(a, b, matrix, 4, 0.1), reference_affine(a, b, matrix, 4, 0.1)):
            self.assert_same_grid(got, expected, "rounding tie")


class BestCells(unittest.TestCase):
    def test_every_maximum_in_row_major_order(self):
        S = [[0, 0, 0], [0, 3, 1], [0, 1, 2], [0, 3, 3]]
        self.assertEqual(find_best_cells(S), (3, [(1, 1), (3, 1), (3, 2)]))

    def test_no_positive_cell(self):
        self.assertEqual(find_best_cells([[0, 0], [0, 0]]), (0, []))

    def test_equal_int_and_float(self):
        best, cells = find_best_cells([[0, 0, 0], [0, 2.5, 3], [0, 3.0, 1]])
        self.assertIs(type(best), int)          # the first maximum found, like max()
        self.assertEqual(cells, [(1, 2), (2, 1)])


class Matrices(unittest.TestCase):
    def test_blosum62_values(self):
        matrix = get_matrix("BLOSUM62")
        self.assertEqual(matrix.kind, PROTEIN)
        self.assertEqual(matrix.score("W", "W"), 11)
        self.assertEqual(matrix.score("A", "A"), 4)
        self.assertEqual(matrix.score("L", "I"), 2)
        self.assertEqual(matrix.score("W", "C"), -2)

    def test_pam250_values(self):
        matrix = get_matrix("PAM250")
        self.assertEqual(matrix.score("W", "W"), 17)
        self.assertEqual(matrix.score("C", "C"), 12)

    def test_all_builtin_matrices_are_symmetric(self):
        names = ["IDENTITY", "TRANSITION_TRANSVERSION", "NUC.4.4", "BLOSUM45", "BLOSUM50",
                 "BLOSUM62", "BLOSUM80", "BLOSUM90", "PAM30", "PAM70", "PAM250"]
        for name in names:
            matrix = get_matrix(name)
            for x in matrix.alphabet:
                for y in matrix.alphabet:
                    self.assertEqual(matrix.score(x, y), matrix.score(y, x), (name, x, y))

    def test_transition_transversion(self):
        matrix = transition_transversion_matrix()
        self.assertEqual(matrix.kind, NUCLEOTIDE)
        self.assertEqual(matrix.score("A", "A"), 1)
        self.assertEqual(matrix.score("A", "G"), -1)     # purine <-> purine
        self.assertEqual(matrix.score("C", "T"), -1)     # pyrimidine <-> pyrimidine
        self.assertEqual(matrix.score("A", "C"), -5)     # purine <-> pyrimidine

    def test_unknown_matrix(self):
        with self.assertRaises(ValueError):
            get_matrix("BLOSUM999")


class Sequences(unittest.TestCase):
    def test_clean_sequence(self):
        self.assertEqual(clean_sequence("  1 acgt acgt\n 11 tt-a "), "ACGTACGTTTA")

    def test_detect_type(self):
        self.assertEqual(detect_type("ACGTTGCA", "GGCAT"), NUCLEOTIDE)
        self.assertEqual(detect_type("ACGUUGCA"), NUCLEOTIDE)
        self.assertEqual(detect_type("HEAGAWGHEE", "ACGT"), PROTEIN)

    def test_rna_is_aligned_as_dna(self):
        self.assertEqual(prepare_sequence("acgu", identity_matrix()), "ACGT")

    def test_invalid_symbol(self):
        with self.assertRaises(ValueError):
            prepare_sequence("ACGTX", identity_matrix())
        with self.assertRaises(ValueError):
            prepare_sequence("   ", identity_matrix())

    def test_read_fasta(self):
        records = read_fasta(EXAMPLES / "dna_pair.fasta")
        self.assertEqual(len(records), 2)
        self.assertTrue(records[0][0].startswith("read_1"))
        header, sequence = read_fasta(EXAMPLES / "hbb_human.fasta")[0]
        self.assertIn("HBB_HUMAN", header)
        self.assertEqual(len(sequence), 147)


class CommandLine(unittest.TestCase):
    @staticmethod
    def run_cli(arguments):
        """Run the CLI with its output captured; return (exit code, stdout)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli_main(arguments)
        return code, out.getvalue()

    def test_runs_on_fasta_files(self):
        code, output = self.run_cli([
            "--fasta1", str(EXAMPLES / "hbb_human.fasta"),
            "--fasta2", str(EXAMPLES / "hba_human.fasta"),
            "--matrix", "BLOSUM62", "--gap", "affine", "--gap-open", "10", "--gap-extend", "1",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Score:       291", output)

    def test_wrong_matrix_for_type_is_an_error(self):
        code, _ = self.run_cli(["--seq1", "ACGT", "--seq2", "ACGT", "--matrix", "BLOSUM62"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
