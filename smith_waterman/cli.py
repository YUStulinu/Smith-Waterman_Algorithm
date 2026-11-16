"""Command line interface.

Examples
--------
    python main.py --seq1 ATCTGA --seq2 TGCA --gap-open 2 --show-matrix
    python main.py --seq1 HEAGAWGHEE --seq2 PAWHEAE --matrix BLOSUM50 --gap-open 8
    python main.py --fasta1 examples/hbb_human.fasta --fasta2 examples/hba_human.fasta \
                   --matrix BLOSUM62 --gap affine --gap-open 10 --gap-extend 1
    python main.py                 (no arguments: interactive mode)
"""

from __future__ import annotations

import argparse
import sys

from .core import smith_waterman
from .formatting import format_result
from .matrices import (
    DESCRIPTIONS,
    NUCLEOTIDE,
    NUCLEOTIDE_MATRICES,
    PROTEIN,
    PROTEIN_MATRICES,
    SubstitutionMatrix,
    default_matrix_name,
    get_matrix,
    load_matrix_file,
)
from .sequences import clean_sequence, detect_type, prepare_sequence, read_fasta, type_label

MAX_MATRIX_CELLS = 40 * 40      # larger DP matrices are not printed in the terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Local alignment of two sequences with the Smith-Waterman algorithm.",
        epilog="Run without arguments to enter the sequences interactively.",
    )
    seqs = parser.add_argument_group("sequences")
    seqs.add_argument("--seq1", help="first sequence, typed directly")
    seqs.add_argument("--seq2", help="second sequence, typed directly")
    seqs.add_argument("--fasta1", help="FASTA file holding the first sequence")
    seqs.add_argument("--fasta2", help="FASTA file holding the second sequence "
                                       "(if omitted, the second record of --fasta1 is used)")
    seqs.add_argument("--type", choices=["auto", "dna", "protein"], default="auto",
                      help="sequence type (default: detected from the symbols)")

    scoring = parser.add_argument_group("scoring")
    scoring.add_argument("--matrix", help="substitution matrix name (see --list-matrices)")
    scoring.add_argument("--matrix-file", help="substitution matrix file in NCBI format")
    scoring.add_argument("--match", type=float, default=1, help="match score for IDENTITY (default 1)")
    scoring.add_argument("--mismatch", type=float, default=-1, help="mismatch score for IDENTITY (default -1)")
    scoring.add_argument("--gap", choices=["linear", "affine"], default="linear",
                         help="gap penalty model (default linear)")
    scoring.add_argument("--gap-open", "-d", type=float, default=None,
                         help="d: penalty per gap (linear) or for opening a gap (affine)")
    scoring.add_argument("--gap-extend", "-e", type=float, default=None,
                         help="e: penalty for extending a gap (affine only)")

    output = parser.add_argument_group("output")
    output.add_argument("--show-matrix", action="store_true", help="print the dynamic programming matrices")
    output.add_argument("--show-substitution", action="store_true", help="print the substitution matrix")
    output.add_argument("--all", action="store_true", help="print every optimal alignment, not only the first")
    output.add_argument("--width", type=int, default=60, help="alignment columns per line (default 60)")
    output.add_argument("--list-matrices", action="store_true", help="list the built-in matrices and exit")
    return parser


def list_matrices() -> str:
    lines = ["Nucleotide matrices:"]
    lines += [f"  {name:<24} {DESCRIPTIONS[name]}" for name in NUCLEOTIDE_MATRICES]
    lines.append("Amino acid matrices:")
    lines += [f"  {name:<24} {DESCRIPTIONS[name]}" for name in PROTEIN_MATRICES]
    return "\n".join(lines)


def _as_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def default_gap_penalties(matrix: SubstitutionMatrix, model: str) -> tuple[int | float, int | float | None]:
    """Sensible defaults: small penalties for unit scores, EMBOSS-like otherwise."""
    large_scores = max(matrix.scores.values()) >= 4
    if model == "affine":
        return (10, 1) if large_scores else (3, 1)
    return (8, None) if large_scores else (2, None)


def choose_matrix(args: argparse.Namespace, kind: str) -> SubstitutionMatrix:
    if args.matrix_file:
        return load_matrix_file(args.matrix_file, kind=kind)
    name = args.matrix or default_matrix_name(kind)
    matrix = get_matrix(name, _as_number(args.match), _as_number(args.mismatch))
    if matrix.kind != kind:
        raise ValueError(
            f"matrix {matrix.name} is a {matrix.kind} matrix but the sequences are {type_label(kind)}; "
            f"use --type or choose another matrix (--list-matrices)"
        )
    return matrix


def load_sequences(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """Return (name_a, raw_a, name_b, raw_b) from the command line options."""
    name_a = name_b = ""
    raw_a = raw_b = None
    if args.fasta1:
        records = read_fasta(args.fasta1)
        name_a, raw_a = records[0]
        if not args.fasta2 and not args.seq2 and len(records) > 1:
            name_b, raw_b = records[1]
    if args.fasta2:
        name_b, raw_b = read_fasta(args.fasta2)[0]
    if args.seq1:
        name_a, raw_a = "seq1", args.seq1
    if args.seq2:
        name_b, raw_b = "seq2", args.seq2
    if raw_a is None or raw_b is None:
        raise ValueError("two sequences are required (--seq1/--seq2 or --fasta1/--fasta2)")
    return name_a, raw_a, name_b, raw_b


def run(args: argparse.Namespace) -> str:
    name_a, raw_a, name_b, raw_b = load_sequences(args)
    if args.type == "auto":
        kind = detect_type(clean_sequence(raw_a), clean_sequence(raw_b))
    else:
        kind = NUCLEOTIDE if args.type == "dna" else PROTEIN

    matrix = choose_matrix(args, kind)
    a = prepare_sequence(raw_a, matrix)
    b = prepare_sequence(raw_b, matrix)

    default_open, default_extend = default_gap_penalties(matrix, args.gap)
    gap_open = _as_number(args.gap_open) if args.gap_open is not None else default_open
    if args.gap == "affine":
        gap_extend = _as_number(args.gap_extend) if args.gap_extend is not None else default_extend
    else:
        if args.gap_extend is not None:
            raise ValueError("--gap-extend is only meaningful together with --gap affine")
        gap_extend = None

    result = smith_waterman(a, b, matrix, gap_open, gap_extend)

    show_matrix = args.show_matrix
    note = ""
    if show_matrix and (len(a) + 1) * (len(b) + 1) > MAX_MATRIX_CELLS:
        show_matrix = False
        note = "\nThe DP matrix is too large to print; use the GUI or shorter sequences.\n"

    report = format_result(result, name_a, name_b, show_matrix, args.all, args.width) + note
    if args.show_substitution:
        report += f"\nSubstitution matrix {matrix.name}:\n{matrix.as_table()}\n"
    return report


def interactive(parser: argparse.ArgumentParser) -> int:
    """Ask for the sequences and parameters one by one."""
    print("Smith-Waterman local alignment - interactive mode")
    print("Press Enter to accept the value shown in [brackets].\n")

    def ask(prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{prompt}{suffix}: ").strip()
        return answer or default

    try:
        seq1 = ask("First sequence", "TGTTACGG")
        seq2 = ask("Second sequence", "GGTTGACTA")
        kind = detect_type(clean_sequence(seq1), clean_sequence(seq2))
        print(f"Detected type: {type_label(kind)}")
        names = NUCLEOTIDE_MATRICES if kind == NUCLEOTIDE else PROTEIN_MATRICES
        print("Available matrices: " + ", ".join(names))
        matrix_name = ask("Substitution matrix", names[0]).upper()

        argv = ["--seq1", seq1, "--seq2", seq2, "--matrix", matrix_name]
        if matrix_name == "IDENTITY":
            argv += ["--match", ask("Match score", "1"), "--mismatch", ask("Mismatch score", "-1")]

        model = ask("Gap penalty model (linear/affine)", "linear").lower()
        argv += ["--gap", model]
        default_open, default_extend = default_gap_penalties(get_matrix(matrix_name), model)
        argv += ["--gap-open", ask("Gap penalty d", str(default_open))]
        if model == "affine":
            argv += ["--gap-extend", ask("Gap extension penalty e", str(default_extend))]
        if ask("Show the DP matrix? (y/n)", "y").lower().startswith("y"):
            argv.append("--show-matrix")

        print()
        print(run(parser.parse_args(argv)))
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return interactive(parser)

    args = parser.parse_args(argv)
    if args.list_matrices:
        print(list_matrices())
        return 0
    try:
        print(run(args))
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
