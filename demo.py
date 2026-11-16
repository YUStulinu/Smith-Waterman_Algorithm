"""Runs the examples discussed in the report, one after another.

Run with:  python demo.py
"""

from __future__ import annotations

from pathlib import Path

from smith_waterman import get_matrix, identity_matrix, read_fasta, smith_waterman
from smith_waterman.formatting import format_result

EXAMPLES = Path(__file__).resolve().parent / "examples"


def show(title: str, a: str, b: str, matrix, gap_open, gap_extend=None, show_matrix=False, show_all=False) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    result = smith_waterman(a, b, matrix, gap_open, gap_extend)
    print(format_result(result, "a", "b", show_matrix=show_matrix, show_all=show_all))


def main() -> None:
    show("Example 1 - course 4: ATCTGA vs TGCA (match 1, mismatch -1, gap 2)",
         "ATCTGA", "TGCA", identity_matrix(1, -1), 2, show_matrix=True)

    show("Example 2 - lab 3-4: TGTTACGG vs GGTTGACTA (match 3, mismatch -3, gap 2)",
         "TGTTACGG", "GGTTGACTA", identity_matrix(3, -3), 2, show_matrix=True)

    dna1 = read_fasta(EXAMPLES / "conserved_region_1.fasta")[0][1]
    dna2 = read_fasta(EXAMPLES / "conserved_region_2.fasta")[0][1]
    show("Example 3 - conserved region hidden in two different sequences (course 4)",
         dna1, dna2, identity_matrix(1, -1), 2)

    show("Example 4 - transitions cheaper than transversions",
         "GATTACAGGT", "CGACTGCAGA", get_matrix("TRANSITION_TRANSVERSION"), 3, show_matrix=True)

    a, b = "ACGTTTGACCATGCA", "ACGTTTCATGCA"
    show("Example 5a - a 3-symbol deletion, LINEAR gaps (d = 3)", a, b, identity_matrix(1, -1), 3, show_all=True)
    show("Example 5b - the same sequences, AFFINE gaps (d = 3, e = 1)", a, b, identity_matrix(1, -1), 3, 1)

    show("Example 6 - proteins, BLOSUM50, linear gap 8 (Durbin et al., 1998)",
         "HEAGAWGHEE", "PAWHEAE", get_matrix("BLOSUM50"), 8, show_matrix=True)

    hbb = read_fasta(EXAMPLES / "hbb_human.fasta")[0][1]
    hba = read_fasta(EXAMPLES / "hba_human.fasta")[0][1]
    show("Example 7a - human hemoglobin beta vs alpha, BLOSUM62, linear gap 8",
         hbb, hba, get_matrix("BLOSUM62"), 8)
    show("Example 7b - human hemoglobin beta vs alpha, BLOSUM62, affine gaps (d = 10, e = 1)",
         hbb, hba, get_matrix("BLOSUM62"), 10, 1)
    show("Example 7c - human hemoglobin beta vs alpha, PAM250, affine gaps (d = 10, e = 1)",
         hbb, hba, get_matrix("PAM250"), 10, 1)


if __name__ == "__main__":
    main()
