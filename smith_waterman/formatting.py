"""Text rendering of alignments and dynamic programming matrices."""

from __future__ import annotations

from .core import AFFINE, NEG_INF, Alignment, Grid, Number, SmithWatermanResult
from .sequences import type_label


def format_number(value: Number) -> str:
    if value == NEG_INF:
        return "-inf"
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def gap_description(result: SmithWatermanResult) -> str:
    if result.gap_model == AFFINE:
        return (
            f"affine, d = {format_number(result.gap_open)} (open), "
            f"e = {format_number(result.gap_extend)} (extend)"
        )
    return f"linear, d = {format_number(result.gap_open)} per gap"


def format_alignment(alignment: Alignment, width: int = 60, name_a: str = "a", name_b: str = "b") -> str:
    """Render an alignment in blocks, with the sequence coordinates.

        a   3 TG-TTACGG   10
              || ||.|
        b   2 TGATTGC      8
    """
    label = max(len(name_a), len(name_b))
    digits = len(str(max(alignment.end_a, alignment.end_b, 1)))
    lines: list[str] = []
    pos_a, pos_b = alignment.start_a, alignment.start_b

    for offset in range(0, alignment.length, width):
        chunk_a = alignment.aligned_a[offset:offset + width]
        chunk_b = alignment.aligned_b[offset:offset + width]
        chunk_mid = alignment.midline[offset:offset + width]
        used_a = len(chunk_a) - chunk_a.count("-")
        used_b = len(chunk_b) - chunk_b.count("-")
        # A block made only of gaps consumes no residue: show the last position.
        last_a = pos_a + used_a - 1 if used_a else pos_a - 1
        last_b = pos_b + used_b - 1 if used_b else pos_b - 1

        lines.append(f"{name_a:<{label}} {pos_a:>{digits}} {chunk_a} {last_a}")
        lines.append(f"{'':<{label}} {'':>{digits}} {chunk_mid}")
        lines.append(f"{name_b:<{label}} {pos_b:>{digits}} {chunk_b} {last_b}")
        lines.append("")
        pos_a += used_a
        pos_b += used_b
    return "\n".join(lines).rstrip()


def format_statistics(alignment: Alignment) -> str:
    length = alignment.length
    aligned_pairs = length - alignment.gaps

    def share(count: int) -> str:
        return f"{count}/{length} ({100 * count / length:.1f}%)"

    return "\n".join([
        f"Score:       {format_number(alignment.score)}",
        f"Length:      {length} columns ({aligned_pairs} aligned pairs)",
        f"Identities:  {share(alignment.identities)}",
        f"Positives:   {share(alignment.positives)}",
        f"Gaps:        {share(alignment.gaps)}",
        f"Region in a: {alignment.start_a}..{alignment.end_a}",
        f"Region in b: {alignment.start_b}..{alignment.end_b}",
    ])


def format_grid(
    grid: Grid, a: str, b: str, path: list[tuple[int, int]] | None = None, title: str = ""
) -> str:
    """Render a DP matrix with sequence a on the rows and b on the columns.

    Cells on the traceback path are wrapped in brackets, e.g. ``[2]``.
    """
    on_path = set(path or [])
    cells = [
        [f"[{format_number(v)}]" if (i, j) in on_path else format_number(v) for j, v in enumerate(row)]
        for i, row in enumerate(grid)
    ]
    width = max(max(len(c) for row in cells for c in row), 1) + 1
    header = "    " + "".join(s.rjust(width) for s in [""] + list(b))
    lines = [title] if title else []
    lines.append(header)
    for i, row in enumerate(cells):
        symbol = a[i - 1] if i > 0 else " "
        lines.append(f"  {symbol} " + "".join(c.rjust(width) for c in row))
    return "\n".join(lines)


def format_result(
    result: SmithWatermanResult,
    name_a: str = "a",
    name_b: str = "b",
    show_matrix: bool = False,
    show_all: bool = False,
    width: int = 60,
) -> str:
    """Full report for one run of the algorithm."""
    lines = [
        "Smith-Waterman local alignment",
        "-" * 62,
        f"Sequence a:   {name_a} ({len(result.a)} symbols)",
        f"Sequence b:   {name_b} ({len(result.b)} symbols)",
        f"Type:         {type_label(result.matrix.kind)}",
        f"Substitution: {result.matrix.name}",
        f"Gap penalty:  {gap_description(result)}",
        "-" * 62,
    ]

    if not result.alignments:
        lines.append("No local alignment with a positive score exists for these parameters.")
    else:
        shown = result.alignments if show_all else result.alignments[:1]
        total = len(result.alignments)
        for index, alignment in enumerate(shown, start=1):
            if total > 1:
                lines.append(f"Optimal alignment {index} of {total} (ends in cell S[{alignment.end_a}][{alignment.end_b}])")
            lines.append(format_statistics(alignment))
            lines.append("")
            lines.append(format_alignment(alignment, width, "a", "b"))
            lines.append("")
        if total > 1 and not show_all:
            lines.append(f"Note: the maximum score is reached in {total} cells; use --all to list every alignment.")

    if show_matrix:
        path = result.alignment.path if result.alignment else None
        lines.append("")
        lines.append(format_grid(result.S, result.a, result.b, path, "Score matrix S (traceback path in brackets):"))
        if result.gap_model == AFFINE:
            lines.append("")
            lines.append(format_grid(result.Gb, result.a, result.b, None, "Matrix Gb (alignment ends with a gap in b):"))
            lines.append("")
            lines.append(format_grid(result.Ga, result.a, result.b, None, "Matrix Ga (alignment ends with a gap in a):"))
    return "\n".join(lines).rstrip() + "\n"
