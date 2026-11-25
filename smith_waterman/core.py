"""Smith-Waterman local alignment with linear and affine gap penalties.

Notation follows course 4 (dynamic programming for sequence alignment):

    a[1..n], b[1..m]   the two sequences
    M(x, y)            substitution score
    d                  gap penalty (linear) / gap opening penalty (affine)
    e                  gap extension penalty (affine), e <= d

Penalties are given as positive numbers and are subtracted from the score.
A run of k consecutive gaps costs

    k * d              with the linear model
    d + (k - 1) * e    with the affine model

Linear gaps -- one matrix S, (n+1) x (m+1):

    S(i, 0) = S(0, j) = 0
    S(i, j) = max{ 0,
                   S(i-1, j-1) + M(a[i], b[j]),     a[i] aligned with b[j]
                   S(i-1, j) - d,                   a[i] aligned with a gap
                   S(i, j-1) - d }                  b[j] aligned with a gap

Affine gaps -- three matrices (Gotoh, 1982):

    S(i, j)   best score of an alignment ending with a[i] aligned with b[j]
    Gb(i, j)  best score of an alignment ending with a gap in b under a[i]
    Ga(i, j)  best score of an alignment ending with a gap in a above b[j]

    S(i, j)  = max{ 0, max{S, Ga, Gb}(i-1, j-1) + M(a[i], b[j]) }
    Gb(i, j) = max{ S(i-1, j) - d, Gb(i-1, j) - e, Ga(i-1, j) - d }
    Ga(i, j) = max{ S(i, j-1) - d, Ga(i, j-1) - e, Gb(i, j-1) - d }

In both cases the alignment ends in the cell holding the maximum of S and the
traceback walks backwards until it reaches a cell of S equal to 0.

Time and memory are O(n * m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import islice

from .matrices import SubstitutionMatrix

NEG_INF = float("-inf")

# Traceback codes. For the linear model they name the move that produced a
# cell; for the affine model they name the matrix the value came from.
# They are recomputed during the traceback, never stored per cell.
STOP = 0
DIAG = 1      # from (i-1, j-1): a[i] aligned with b[j]      / matrix S
UP = 2        # from (i-1, j):   a[i] aligned with a gap     / matrix Gb
LEFT = 3      # from (i, j-1):   b[j] aligned with a gap     / matrix Ga

LINEAR = "linear"
AFFINE = "affine"

Number = int | float
Grid = list[list[Number]]


@dataclass
class Alignment:
    """One local alignment produced by the traceback."""

    aligned_a: str
    aligned_b: str
    score: Number
    start_a: int            # 1-based position in a of the first aligned symbol
    end_a: int              # 1-based position in a of the last aligned symbol
    start_b: int
    end_b: int
    path: list[tuple[int, int]]   # visited cells (i, j), from start to end
    matrix: SubstitutionMatrix = field(repr=False)

    @property
    def length(self) -> int:
        return len(self.aligned_a)

    @property
    def identities(self) -> int:
        return sum(x == y for x, y in zip(self.aligned_a, self.aligned_b))

    @property
    def positives(self) -> int:
        """Columns with a positive substitution score (similar residues)."""
        return sum(
            x != "-" and y != "-" and self.matrix.score(x, y) > 0
            for x, y in zip(self.aligned_a, self.aligned_b)
        )

    @property
    def gaps(self) -> int:
        return self.aligned_a.count("-") + self.aligned_b.count("-")

    @property
    def midline(self) -> str:
        """Match line: '|' identical, ':' similar, '.' mismatch, ' ' gap."""
        marks = []
        for x, y in zip(self.aligned_a, self.aligned_b):
            if x == "-" or y == "-":
                marks.append(" ")
            elif x == y:
                marks.append("|")
            elif self.matrix.score(x, y) > 0:
                marks.append(":")
            else:
                marks.append(".")
        return "".join(marks)


@dataclass
class SmithWatermanResult:
    """Everything computed for a pair of sequences."""

    a: str
    b: str
    matrix: SubstitutionMatrix
    gap_model: str                  # LINEAR or AFFINE
    gap_open: Number                # d
    gap_extend: Number              # e (equal to d for the linear model)
    S: Grid
    Ga: Grid | None                 # only for the affine model
    Gb: Grid | None
    best_score: Number
    alignments: list[Alignment]           # one alignment per cell of S holding the maximum

    @property
    def alignment(self) -> Alignment | None:
        """The first optimal alignment (None if the best score is 0)."""
        return self.alignments[0] if self.alignments else None


def _check_penalties(d: Number, e: Number | None) -> None:
    if not math.isfinite(d) or (e is not None and not math.isfinite(e)):
        raise ValueError("gap penalties must be finite numbers")
    if d < 0 or (e is not None and e < 0):
        raise ValueError("gap penalties must be given as non-negative numbers")
    if e is not None and e > d:
        raise ValueError("the gap extension penalty e must not exceed the opening penalty d")


def _substitution_rows(a: str, b: str, matrix: SubstitutionMatrix) -> list[list[Number]]:
    """Row i holds M(a[i], b[j]) for every j.

    Rows are shared between equal symbols of a, so at most one row per letter
    of the alphabet is built instead of n * m scores.
    """
    cache: dict[str, list[Number]] = {}
    for x in set(a):
        cache[x] = [matrix.score(x, y) for y in b]
    return [cache[x] for x in a]


# The fill loops below walk the previous row with zip() instead of indexing
# S[i][j], and compare values directly instead of calling max(): this is
# several times faster in CPython. They store only the scores. The traceback
# re-runs the same comparisons for the few cells on its path (_linear_move,
# _affine_*_move), which gives exactly the moves the fill chose, without
# keeping an (n+1) x (m+1) traceback matrix in memory.

# --------------------------------------------------------------------------
# Linear gap penalty
# --------------------------------------------------------------------------

def fill_linear(a: str, b: str, matrix: SubstitutionMatrix, d: Number) -> Grid:
    """Fill the score matrix S for linear gaps."""
    S: Grid = [[0] * (len(b) + 1)]
    previous = S[0]

    for sub in _substitution_rows(a, b, matrix):
        row = [0]
        left = 0                                # S(i, j-1)
        # diag = S(i-1, j-1), up = S(i-1, j), s = M(a[i], b[j])
        for diag, up, s in zip(previous, islice(previous, 1, None), sub):
            # Same priority as the pseudocode in the course: diagonal, up, left.
            best = diag + s
            up -= d
            if up > best:
                best = up
            left -= d
            if left > best:
                best = left
            if best <= 0:
                best = 0
            row.append(best)
            left = best
        S.append(row)
        previous = row
    return S


def _linear_move(a: str, b: str, matrix: SubstitutionMatrix, S: Grid, d: Number, i: int, j: int) -> int:
    """The move fill_linear chose for cell (i, j)."""
    if S[i][j] <= 0:                    # reset to 0 (this includes row 0 and column 0)
        return STOP
    best = S[i - 1][j - 1] + matrix.score(a[i - 1], b[j - 1])
    move = DIAG
    if S[i - 1][j] - d > best:
        best, move = S[i - 1][j] - d, UP
    if S[i][j - 1] - d > best:
        move = LEFT
    return move


def traceback_linear(
    a: str, b: str, matrix: SubstitutionMatrix, S: Grid, d: Number, end: tuple[int, int]
) -> Alignment:
    """Rebuild the alignment that ends in cell ``end`` of S."""
    i, j = end
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    path = [(i, j)]

    while (move := _linear_move(a, b, matrix, S, d, i, j)) != STOP:
        if move == DIAG:
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            i, j = i - 1, j - 1
        elif move == UP:
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            j -= 1
        path.append((i, j))

    path.reverse()
    return Alignment(
        aligned_a="".join(reversed(aligned_a)),
        aligned_b="".join(reversed(aligned_b)),
        score=S[end[0]][end[1]],
        start_a=i + 1, end_a=end[0],
        start_b=j + 1, end_b=end[1],
        path=path,
        matrix=matrix,
    )


# --------------------------------------------------------------------------
# Affine gap penalty (three matrices)
# --------------------------------------------------------------------------

def fill_affine(a: str, b: str, matrix: SubstitutionMatrix, d: Number, e: Number) -> tuple[Grid, Grid, Grid]:
    """Fill the three matrices S, Ga and Gb for affine gaps."""
    m = len(b)
    S: Grid = [[0] * (m + 1)]
    Ga: Grid = [[NEG_INF] * (m + 1)]
    Gb: Grid = [[NEG_INF] * (m + 1)]
    prev_S, prev_Ga, prev_Gb = S[0], Ga[0], Gb[0]

    for sub in _substitution_rows(a, b, matrix):
        row_S, row_Ga, row_Gb = [0], [NEG_INF], [NEG_INF]
        s_left, ga_left, gb_left = 0, NEG_INF, NEG_INF          # column j-1 of row i
        # *_diag = column j-1 of row i-1, *_up = column j of row i-1
        for s_diag, s_up, ga_diag, ga_up, gb_diag, gb_up, s in zip(
            prev_S, islice(prev_S, 1, None),
            prev_Ga, islice(prev_Ga, 1, None),
            prev_Gb, islice(prev_Gb, 1, None),
            sub,
        ):
            # Gb(i, j): a[i] is aligned with a gap, we come from row i-1.
            # Priority on ties: open, extend, switch (same for Ga).
            gb = s_up - d
            value = gb_up - e
            if value > gb:
                gb = value
            value = ga_up - d
            if value > gb:
                gb = value

            # Ga(i, j): b[j] is aligned with a gap, we come from column j-1.
            ga = s_left - d
            value = ga_left - e
            if value > ga:
                ga = value
            value = gb_left - d
            if value > ga:
                ga = value

            # S(i, j): a[i] is aligned with b[j], we come from (i-1, j-1).
            # Priority on ties: S, Gb, Ga.
            best = s_diag
            if gb_diag > best:
                best = gb_diag
            if ga_diag > best:
                best = ga_diag
            best += s
            if best <= 0:
                best = 0

            row_S.append(best)
            row_Ga.append(ga)
            row_Gb.append(gb)
            s_left, ga_left, gb_left = best, ga, gb

        S.append(row_S)
        Ga.append(row_Ga)
        Gb.append(row_Gb)
        prev_S, prev_Ga, prev_Gb = row_S, row_Ga, row_Gb
    return S, Ga, Gb


def _best_of(first: Number, second: Number, third: Number, d1: Number, d2: Number, d3: Number) -> int:
    """Which of first - d1, second - d2, third - d3 the fill kept (DIAG, UP, LEFT).

    Ties go to the earliest candidate, exactly as in fill_affine.
    """
    best, move = first - d1, DIAG
    if second - d2 > best:
        best, move = second - d2, UP
    if third - d3 > best:
        move = LEFT
    return move


def traceback_affine(
    a: str,
    b: str,
    matrix: SubstitutionMatrix,
    S: Grid,
    Ga: Grid,
    Gb: Grid,
    d: Number,
    e: Number,
    end: tuple[int, int],
) -> Alignment:
    """Rebuild the alignment that ends in cell ``end`` of S.

    The traceback keeps track of the current matrix (DIAG = S, UP = Gb,
    LEFT = Ga). The current matrix decides which column is emitted; the
    candidate its recurrence kept decides which matrix becomes current next.
    """
    i, j = end
    state = DIAG                       # a local alignment always ends in S
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    path = [(i, j)]

    while not (state == DIAG and S[i][j] <= 0):
        if state == DIAG:
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            i, j = i - 1, j - 1
            # S(i+1, j+1) = max{S, Gb, Ga}(i, j) + M: ties go to S, then Gb.
            state = _best_of(S[i][j], Gb[i][j], Ga[i][j], 0, 0, 0)
        elif state == UP:              # matrix Gb: gap in b
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            # Gb candidates in fill order: open (S), extend (Gb), switch (Ga).
            state = _best_of(S[i - 1][j], Gb[i - 1][j], Ga[i - 1][j], d, e, d)
            i -= 1
        else:                          # matrix Ga: gap in a
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            # Ga candidates in fill order: open (S), extend (Ga), switch (Gb).
            move = _best_of(S[i][j - 1], Ga[i][j - 1], Gb[i][j - 1], d, e, d)
            state = {DIAG: DIAG, UP: LEFT, LEFT: UP}[move]
            j -= 1
        path.append((i, j))

    path.reverse()
    return Alignment(
        aligned_a="".join(reversed(aligned_a)),
        aligned_b="".join(reversed(aligned_b)),
        score=S[end[0]][end[1]],
        start_a=i + 1, end_a=end[0],
        start_b=j + 1, end_b=end[1],
        path=path,
        matrix=matrix,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def find_best_cells(S: Grid) -> tuple[Number, list[tuple[int, int]]]:
    """Return the maximum of S and every cell where it is reached."""
    best = max(max(row) for row in S)
    if best <= 0:
        return 0, []
    cells = [(i, j) for i, row in enumerate(S) for j, value in enumerate(row) if value == best]
    return best, cells


def smith_waterman(
    a: str,
    b: str,
    matrix: SubstitutionMatrix,
    gap_open: Number = 2,
    gap_extend: Number | None = None,
) -> SmithWatermanResult:
    """Compute the optimal local alignment(s) of ``a`` and ``b``.

    With ``gap_extend=None`` the linear gap model is used (every gap costs
    ``gap_open``). Otherwise the affine model is used: the first gap of a run
    costs ``gap_open`` and each following gap costs ``gap_extend``.

    The sequences must already be upper-case and contain only symbols of the
    matrix alphabet (see ``sequences.prepare_sequence``).
    """
    if not a or not b:
        raise ValueError("both sequences must be non-empty")
    _check_penalties(gap_open, gap_extend)

    if gap_extend is None:
        S = fill_linear(a, b, matrix, gap_open)
        best, cells = find_best_cells(S)
        # A local alignment ends with a pair of symbols. With d = 0 a gap is
        # free, so the maximum can also sit in cells reached by trailing gaps;
        # those are the same alignment plus useless gaps and are dropped.
        alignments = [
            traceback_linear(a, b, matrix, S, gap_open, (i, j))
            for i, j in cells
            if _linear_move(a, b, matrix, S, gap_open, i, j) == DIAG
        ]
        return SmithWatermanResult(a, b, matrix, LINEAR, gap_open, gap_open, S, None, None, best, alignments)

    S, Ga, Gb = fill_affine(a, b, matrix, gap_open, gap_extend)
    best, cells = find_best_cells(S)
    alignments = [traceback_affine(a, b, matrix, S, Ga, Gb, gap_open, gap_extend, cell) for cell in cells]
    return SmithWatermanResult(a, b, matrix, AFFINE, gap_open, gap_extend, S, Ga, Gb, best, alignments)


def score_alignment(
    aligned_a: str,
    aligned_b: str,
    matrix: SubstitutionMatrix,
    gap_open: Number,
    gap_extend: Number | None = None,
) -> Number:
    """Score an already built alignment directly from its columns.

    Independent of the dynamic programming code, so it is used to verify that
    the alignment returned by the traceback really has the reported score.
    """
    extend = gap_open if gap_extend is None else gap_extend
    total: Number = 0
    previous = None                     # 'a' / 'b' if the previous column was a gap
    for x, y in zip(aligned_a, aligned_b, strict=True):
        if x == "-" and y == "-":
            raise ValueError("a column cannot contain two gaps")
        if x == "-" or y == "-":
            current = "a" if x == "-" else "b"
            total -= extend if current == previous else gap_open
            previous = current
        else:
            total += matrix.score(x, y)
            previous = None
    return total
