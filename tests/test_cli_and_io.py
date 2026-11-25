"""Functional tests for the command line, the input files and the text output.

Run from the project folder with:  python -m unittest -v
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smith_waterman import (
    NUCLEOTIDE,
    PROTEIN,
    available_matrices,
    get_matrix,
    identity_matrix,
    load_matrix_file,
    read_fasta,
    score_alignment,
    smith_waterman,
)
from smith_waterman import cli
from smith_waterman.formatting import format_alignment, format_grid, format_number, format_result

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_cli(arguments, stdin_lines=None):
    """Run the CLI with captured output; return (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    answers = iter(stdin_lines or [])

    def fake_input(prompt=""):
        out.write(prompt)
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            mock.patch("builtins.input", fake_input):
        code = cli.main(arguments)
    return code, out.getvalue(), err.getvalue()


class TempFiles(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = Path(self.folder.name) / name
        path.write_text(text, encoding=encoding)
        return path


class CommandLineOptions(unittest.TestCase):
    def test_help_lists_every_option(self):
        with self.assertRaises(SystemExit) as exit_, contextlib.redirect_stdout(io.StringIO()) as out:
            cli.main(["--help"])
        self.assertEqual(exit_.exception.code, 0)
        for option in ("--seq1", "--fasta1", "--matrix-file", "--gap-extend", "--show-matrix", "--width"):
            self.assertIn(option, out.getvalue())

    def test_list_matrices(self):
        code, output, _ = run_cli(["--list-matrices"])
        self.assertEqual(code, 0)
        for name in available_matrices(NUCLEOTIDE) + available_matrices(PROTEIN):
            self.assertIn(name, output)

    def test_width_must_be_positive(self):
        self.assertEqual(cli.positive_int("5"), 5)
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.positive_int("0")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["--seq1", "ACGT", "--seq2", "ACGT", "--width", "-3"])

    def test_show_matrix_prints_the_course_matrix(self):
        code, output, _ = run_cli(["--seq1", "ATCTGA", "--seq2", "TGCA", "--gap-open", "2", "--show-matrix"])
        self.assertEqual(code, 0)
        self.assertIn("Score matrix S", output)
        self.assertIn("[2]", output)                     # the end of the traceback path
        self.assertIn("Score:       2", output)

    def test_show_matrix_affine_prints_three_matrices(self):
        code, output, _ = run_cli(["--seq1", "ACGTTTGACCATGCA", "--seq2", "ACGTTTCATGCA",
                                   "--gap", "affine", "-d", "3", "-e", "1", "--show-matrix"])
        self.assertEqual(code, 0)
        self.assertIn("Matrix Gb", output)
        self.assertIn("Matrix Ga", output)
        self.assertIn("-inf", output)
        self.assertIn("affine, d = 3 (open), e = 1 (extend)", output)

    def test_large_matrix_is_not_printed(self):
        code, output, _ = run_cli(["--seq1", "A" * 50, "--seq2", "A" * 50, "--show-matrix"])
        self.assertEqual(code, 0)
        self.assertIn("too large to print", output)
        self.assertNotIn("Score matrix S", output)

    def test_show_substitution_and_all(self):
        code, output, _ = run_cli(["--seq1", "ACGTTACG", "--seq2", "ACG", "--show-substitution", "--all"])
        self.assertEqual(code, 0)
        self.assertIn("Optimal alignment 1 of 2", output)
        self.assertIn("Optimal alignment 2 of 2", output)
        self.assertIn("Substitution matrix IDENTITY(1/-1)", output)

    def test_note_when_several_alignments_are_hidden(self):
        _, output, _ = run_cli(["--seq1", "ACGTTACG", "--seq2", "ACG"])
        self.assertIn("use --all", output)

    def test_no_alignment(self):
        code, output, _ = run_cli(["--seq1", "AAAA", "--seq2", "TTTT"])
        self.assertEqual(code, 0)
        self.assertIn("No local alignment with a positive score", output)

    def test_custom_match_and_mismatch(self):
        _, output, _ = run_cli(["--seq1", "TGTTACGG", "--seq2", "GGTTGACTA",
                                "--match", "3", "--mismatch", "-3", "-d", "2"])
        self.assertIn("Score:       13", output)

    def test_forced_type(self):
        code, output, _ = run_cli(["--seq1", "ACGT", "--seq2", "ACGT", "--type", "protein"])
        self.assertEqual(code, 0)
        self.assertIn("BLOSUM62", output)
        code, output, _ = run_cli(["--seq1", "ACGT", "--seq2", "ACGT", "--type", "dna"])
        self.assertEqual(code, 0)
        self.assertIn("nucleotide", output)

    def test_default_penalties_follow_the_matrix_scale(self):
        self.assertEqual(cli.default_gap_penalties(identity_matrix(), "linear"), (2, None))
        self.assertEqual(cli.default_gap_penalties(identity_matrix(), "affine"), (3, 1))
        self.assertEqual(cli.default_gap_penalties(get_matrix("BLOSUM62"), "linear"), (8, None))
        self.assertEqual(cli.default_gap_penalties(get_matrix("BLOSUM62"), "affine"), (10, 1))
        _, output, _ = run_cli(["--seq1", "HEAGAWGHEE", "--seq2", "PAWHEAE", "--gap", "affine"])
        self.assertIn("affine, d = 10 (open), e = 1 (extend)", output)

    def test_fractional_penalties_are_kept(self):
        _, output, _ = run_cli(["--seq1", "ACGTACGT", "--seq2", "ACGACGT", "-d", "1.5"])
        self.assertIn("linear, d = 1.5 per gap", output)


class CommandLineErrors(unittest.TestCase):
    def assert_error(self, arguments, message):
        code, _, error = run_cli(arguments)
        self.assertEqual(code, 1)
        self.assertIn(message, error)

    def test_missing_second_sequence(self):
        self.assert_error(["--seq1", "ACGT"], "two sequences are required")

    def test_gap_extend_needs_affine(self):
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "-e", "1"], "only meaningful together with --gap affine")

    def test_negative_or_infinite_penalties(self):
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "-d", "-1"], "non-negative")
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "-d", "inf"], "finite")
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "-d", "nan"], "finite")
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "--gap", "affine", "-d", "1", "-e", "2"],
                          "must not exceed")

    def test_invalid_scores_for_identity(self):
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "--match", "0"], "match score must be positive")
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "--mismatch", "1"], "mismatch score must be negative")

    def test_symbol_outside_the_alphabet(self):
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGTN", "--type", "dna"], "cannot be scored")

    def test_unknown_matrix(self):
        self.assert_error(["--seq1", "ACGT", "--seq2", "ACGT", "--matrix", "NOPE"], "unknown substitution matrix")

    def test_missing_file(self):
        code, _, error = run_cli(["--fasta1", "does_not_exist.fasta", "--seq2", "ACGT"])
        self.assertEqual(code, 1)
        self.assertIn("Error:", error)


class FastaInput(TempFiles):
    def test_two_records_in_one_file(self):
        code, output, _ = run_cli(["--fasta1", str(EXAMPLES / "dna_pair.fasta")])
        self.assertEqual(code, 0)
        self.assertIn("Sequence a:   read_1", output)
        self.assertIn("Sequence b:   read_2", output)

    def test_seq_options_override_files(self):
        code, output, _ = run_cli(["--fasta1", str(EXAMPLES / "dna_pair.fasta"), "--seq2", "ACGT"])
        self.assertEqual(code, 0)
        self.assertIn("Sequence b:   seq2 (4 symbols)", output)

    def test_file_without_header_comments_and_bom(self):
        path = self.write("raw.txt", "﻿; a comment\n1 acgtac\n\n11 gtt\n", encoding="utf-8")
        self.assertEqual(read_fasta(path), [("raw", "ACGTACGTT")])

    def test_empty_file(self):
        with self.assertRaises(ValueError):
            read_fasta(self.write("empty.fasta", "\n; only a comment\n"))

    def test_several_records(self):
        path = self.write("three.fasta", ">x first\nAC\nGT\n>y\nTT\n>z\n")
        self.assertEqual(read_fasta(path), [("x first", "ACGT"), ("y", "TT"), ("z", "")])


class MatrixFiles(TempFiles):
    NUC = "# comment\n   A  C  G  T\nA  2 -1 -1 -1\nC -1  2 -1 -1\nG -1 -1  2 -1\nT -1 -1 -1  2\n"

    def test_load_and_use(self):
        path = self.write("my_dna.txt", self.NUC)
        matrix = load_matrix_file(path)
        self.assertEqual((matrix.name, matrix.kind, matrix.score("A", "A")), ("my_dna", NUCLEOTIDE, 2))
        code, output, _ = run_cli(["--seq1", "ACGT", "--seq2", "ACGT", "--matrix-file", str(path)])
        self.assertEqual(code, 0)
        self.assertIn("Score:       8", output)

    def test_fractional_scores_and_bom(self):
        path = self.write("half.txt", "﻿" + self.NUC.replace(" 2", "1.5"))
        matrix = load_matrix_file(path)
        self.assertEqual(matrix.score("C", "C"), 1.5)
        self.assertIn("1.5", matrix.as_table())

    def test_kind_is_detected_from_the_alphabet(self):
        self.assertEqual(load_matrix_file(Path(cli.__file__).parent / "data" / "BLOSUM62.txt").kind, PROTEIN)

    def test_invalid_files(self):
        cases = {
            "not a substitution matrix": "A C G T\n",
            "has 3 scores": self.NUC.replace("C -1  2 -1 -1", "C -1  2 -1"),
            "not a finite number": self.NUC.replace("G -1 -1  2 -1", "G -1 -1 inf -1"),
            "symbols differ": self.NUC.replace("T -1 -1 -1  2", "U -1 -1 -1  2"),
            "not symmetric": self.NUC.replace("A  2 -1 -1 -1", "A  2 -3 -1 -1"),
        }
        for message, text in cases.items():
            with self.subTest(message), self.assertRaises(ValueError) as error:
                load_matrix_file(self.write("bad.txt", text))
            self.assertIn(message, str(error.exception))

    def test_identity_scores_must_be_finite(self):
        with self.assertRaises(ValueError):
            identity_matrix(float("inf"), -1)
        with self.assertRaises(ValueError):
            identity_matrix(1, float("nan"))

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            available_matrices("lipid")


class InteractiveMode(unittest.TestCase):
    def test_defaults(self):
        # sequence a, b, matrix, match, mismatch, gap model, d, show matrix
        code, output, _ = run_cli([], [""] * 8)
        self.assertEqual(code, 0)
        self.assertIn("Detected type: nucleotide", output)
        self.assertIn("Score matrix S", output)

    def test_identity_with_custom_scores_and_affine(self):
        code, output, _ = run_cli([], ["TGTTACGG", "GGTTGACTA", "identity", "3", "-3", "affine", "4", "1", "n"])
        self.assertEqual(code, 0)
        self.assertIn("IDENTITY(3/-3)", output)
        self.assertIn("affine, d = 4 (open), e = 1 (extend)", output)
        self.assertNotIn("Score matrix S", output)

    def test_protein(self):
        code, output, _ = run_cli([], ["HEAGAWGHEE", "PAWHEAE", "BLOSUM50", "linear", "8", "n"])
        self.assertEqual(code, 0)
        self.assertIn("Detected type: protein", output)
        self.assertIn("Score:       28", output)

    def test_invalid_answer(self):
        code, _, error = run_cli([], ["ACGT", "ACGT", "BLOSUM62", "linear", "8", "n"])
        self.assertEqual(code, 1)
        self.assertIn("Error:", error)

    def test_end_of_input(self):
        code, _, _ = run_cli([], ["ACGT"])
        self.assertEqual(code, 1)


class TextOutput(unittest.TestCase):
    def test_format_number(self):
        self.assertEqual(format_number(3.0), "3")
        self.assertEqual(format_number(2.5), "2.5")
        self.assertEqual(format_number(float("-inf")), "-inf")

    def test_alignment_blocks_and_coordinates(self):
        alignment = smith_waterman("ACGTTTGACCATGCA", "ACGTTTCATGCA", identity_matrix(), 3, 1).alignment
        text = format_alignment(alignment, width=3, name_a="x", name_b="y")
        lines = text.splitlines()
        # The third block of b holds only gaps: it consumes no residue.
        self.assertEqual(lines[8], "x  7 GAC 9")
        self.assertEqual(lines[10], "y  7 --- 6")
        self.assertEqual(lines[14], "y  7 CAT 9")

    def test_midline_marks(self):
        alignment = smith_waterman("HEAGAWGHEE", "PAWHEAE", get_matrix("BLOSUM50"), 8).alignment
        self.assertEqual(alignment.midline, "|| ||")
        alignment = smith_waterman("LLLKV", "LLIKV", get_matrix("BLOSUM62"), 8).alignment
        self.assertEqual(alignment.midline, "||:||")
        self.assertEqual((alignment.identities, alignment.positives, alignment.gaps), (4, 5, 0))
        alignment = smith_waterman("ACGTACGT", "ACGAACGT", identity_matrix(), 2).alignment
        self.assertEqual(alignment.midline, "|||.||||")

    def test_grid_with_path(self):
        result = smith_waterman("AC", "AC", identity_matrix(), 1)
        grid = format_grid(result.S, result.a, result.b, result.alignment.path, "S:")
        self.assertEqual(grid.splitlines()[0], "S:")
        self.assertIn("[2]", grid)
        self.assertIn("[0]", grid)

    def test_report_without_alignment_and_with_matrix(self):
        result = smith_waterman("AA", "TT", identity_matrix(), 1)
        report = format_result(result, show_matrix=True)
        self.assertIn("No local alignment", report)
        self.assertIn("Score matrix S", report)
        self.assertNotIn("[", report)


class CoreEdgeCases(unittest.TestCase):
    def test_empty_sequences(self):
        with self.assertRaises(ValueError):
            smith_waterman("", "ACGT", identity_matrix())

    def test_non_finite_extension(self):
        with self.assertRaises(ValueError):
            smith_waterman("ACGT", "ACGT", identity_matrix(), 2, float("nan"))
        with self.assertRaises(ValueError):
            smith_waterman("ACGT", "ACGT", identity_matrix(), 2, -1)

    def test_score_alignment(self):
        matrix = identity_matrix(1, -1)
        self.assertEqual(score_alignment("AC--GT", "ACTTGT", matrix, 3, 1), 4 - 4)
        self.assertEqual(score_alignment("AC-G-T", "ACT-TT", matrix, 3), 1 + 1 - 3 - 3 - 3 + 1)
        with self.assertRaises(ValueError):
            score_alignment("A-", "A-", matrix, 1)
        with self.assertRaises(ValueError):
            score_alignment("AC", "A", matrix, 1)

    def test_free_gaps_drop_trailing_gap_cells(self):
        """With d = 0 every maximum cell reached by a gap is the same alignment plus gaps."""
        result = smith_waterman("ACGT", "ACGTAAA", identity_matrix(), 0)
        self.assertEqual(result.best_score, 4)
        for alignment in result.alignments:
            self.assertNotEqual(alignment.aligned_b[-1], "-")
            self.assertNotEqual(alignment.aligned_a[-1], "-")

    def test_affine_switching_between_gap_matrices(self):
        """A gap in a followed directly by a gap in b is scored as two openings."""
        matrix = identity_matrix(5, -4)
        result = smith_waterman("AAAAACGGGGG", "AAAAATGGGGG", matrix, 3, 1)
        alignment = result.alignment
        self.assertEqual(score_alignment(alignment.aligned_a, alignment.aligned_b, matrix, 3, 1), result.best_score)

    def test_ties_follow_the_course_priority(self):
        """Linear: same alignment as the course pseudocode (full S matrix, ties: diagonal, up, left)."""

        def course_pseudocode(a, b, matrix, d):
            n, m = len(a), len(b)
            S = [[0] * (m + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    S[i][j] = max(0, S[i - 1][j - 1] + matrix.score(a[i - 1], b[j - 1]),
                                  S[i - 1][j] - d, S[i][j - 1] - d)
            best = max(max(row) for row in S)
            i, j = next((i, j) for i in range(n + 1) for j in range(m + 1) if S[i][j] == best)
            x, y = "", ""
            while S[i][j] > 0:
                if S[i][j] == S[i - 1][j - 1] + matrix.score(a[i - 1], b[j - 1]):
                    x, y, i, j = a[i - 1] + x, b[j - 1] + y, i - 1, j - 1
                elif S[i][j] == S[i - 1][j] - d:
                    x, y, i = a[i - 1] + x, "-" + y, i - 1
                else:
                    x, y, j = "-" + x, b[j - 1] + y, j - 1
            return best, x, y

        rng = random.Random(11)
        matrix = identity_matrix(2, -1)
        for _ in range(300):
            a = "".join(rng.choice("ACG") for _ in range(rng.randint(1, 15)))
            b = "".join(rng.choice("ACG") for _ in range(rng.randint(1, 15)))
            d = rng.choice([1, 2, 3])
            result = smith_waterman(a, b, matrix, d)
            if result.alignment is None:
                continue
            got = (result.best_score, result.alignment.aligned_a, result.alignment.aligned_b)
            self.assertEqual(got, course_pseudocode(a, b, matrix, d), (a, b, d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
