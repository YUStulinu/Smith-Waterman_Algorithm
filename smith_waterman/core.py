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

from dataclasses import dataclass, field

from .matrices import SubstitutionMatrix

NEG_INF = float("-inf")

# Traceback codes. For the linear model they name the move that produced a
# cell; for the affine model they name the matrix the value came from.
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
    best_cells: list[tuple[int, int]]     # every cell of S holding the maximum
    alignments: list[Alignment]           # one alignment per best cell

    @property
    def alignment(self) -> Alignment | None:
        """The first optimal alignment (None if the best score is 0)."""
        return self.alignments[0] if self.alignments else None


def _check_penalties(d: Number, e: Number | None) -> None:
    if d < 0 or (e is not None and e < 0):
        raise ValueError("gap penalties must be given as non-negative numbers")
    if e is not None and e > d:
        raise ValueError("the gap extension penalty e must not exceed the opening penalty d")


def _substitution_rows(a: str, b: str, matrix: SubstitutionMatrix) -> list[list[Number]]:
    """Pre-compute M(a[i], b[j]) for every pair, to keep the main loops simple."""
    return [[matrix.score(x, y) for y in b] for x in a]


# --------------------------------------------------------------------------
# Linear gap penalty
# --------------------------------------------------------------------------

def fill_linear(a: str, b: str, matrix: SubstitutionMatrix, d: Number) -> tuple[Grid, list[list[int]]]:
    """Fill the score matrix S and the traceback matrix for linear gaps."""
    n, m = len(a), len(b)
    sub = _substitution_rows(a, b, matrix)
    S: Grid = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[STOP] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = S[i - 1][j - 1] + sub[i - 1][j - 1]
            up = S[i - 1][j] - d
            left = S[i][j - 1] - d
            best = max(0, diagonal, up, left)
            S[i][j] = best
            # Same priority as the pseudocode in the course: diagonal, up, left.
            if best == 0:
                back[i][j] = STOP
            elif best == diagonal:
                back[i][j] = DIAG
            elif best == up:
                back[i][j] = UP
            else:
                back[i][j] = LEFT
    return S, back


def traceback_linear(
    a: str, b: str, matrix: SubstitutionMatrix, S: Grid, back: list[list[int]], end: tuple[int, int]
) -> Alignment:
    """Rebuild the alignment that ends in cell ``end`` of S."""
    i, j = end
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    path = [(i, j)]

    while back[i][j] != STOP:
        move = back[i][j]
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

def fill_affine(
    a: str, b: str, matrix: SubstitutionMatrix, d: Number, e: Number
) -> tuple[Grid, Grid, Grid, list[list[int]], list[list[int]], list[list[int]]]:
    """Fill S, Ga, Gb and one traceback matrix for each of them.

    Each traceback entry stores the matrix (DIAG = S, UP = Gb, LEFT = Ga) the
    value was derived from, or STOP for a cell of S that was reset to 0.
    """
    n, m = len(a), len(b)
    sub = _substitution_rows(a, b, matrix)

    S: Grid = [[0] * (m + 1) for _ in range(n + 1)]
    Ga: Grid = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Gb: Grid = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    back_S = [[STOP] * (m + 1) for _ in range(n + 1)]
    back_Ga = [[STOP] * (m + 1) for _ in range(n + 1)]
    back_Gb = [[STOP] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Gb(i, j): a[i] is aligned with a gap, we come from row i-1.
            open_gap = S[i - 1][j] - d
            extend_gap = Gb[i - 1][j] - e
            switch_gap = Ga[i - 1][j] - d
            best = max(open_gap, extend_gap, switch_gap)
            Gb[i][j] = best
            back_Gb[i][j] = DIAG if best == open_gap else UP if best == extend_gap else LEFT

            # Ga(i, j): b[j] is aligned with a gap, we come from column j-1.
            open_gap = S[i][j - 1] - d
            extend_gap = Ga[i][j - 1] - e
            switch_gap = Gb[i][j - 1] - d
            best = max(open_gap, extend_gap, switch_gap)
            Ga[i][j] = best
            back_Ga[i][j] = DIAG if best == open_gap else LEFT if best == extend_gap else UP

            # S(i, j): a[i] is aligned with b[j], we come from (i-1, j-1).
            from_S = S[i - 1][j - 1]
            from_Gb = Gb[i - 1][j - 1]
            from_Ga = Ga[i - 1][j - 1]
            previous = max(from_S, from_Gb, from_Ga)
            best = max(0, previous + sub[i - 1][j - 1])
            S[i][j] = best
            if best == 0:
                back_S[i][j] = STOP
            elif previous == from_S:
                back_S[i][j] = DIAG
            elif previous == from_Gb:
                back_S[i][j] = UP
            else:
                back_S[i][j] = LEFT
    return S, Ga, Gb, back_S, back_Ga, back_Gb


def traceback_affine(
    a: str,
    b: str,
    matrix: SubstitutionMatrix,
    S: Grid,
    back_S: list[list[int]],
    back_Ga: list[list[int]],
    back_Gb: list[list[int]],
    end: tuple[int, int],
) -> Alignment:
    """Rebuild the alignment that ends in cell ``end`` of S.

    The traceback keeps track of the current matrix. The current matrix
    decides which column is emitted, and its traceback entry decides which
    matrix becomes current after the move.
    """
    i, j = end
    state = DIAG                       # a local alignment always ends in S
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    path = [(i, j)]

    while not (state == DIAG and back_S[i][j] == STOP):
        if state == DIAG:
            aligned_a.append(a[i - 1])
            aligned_b.append(b[j - 1])
            state = back_S[i][j]
            i, j = i - 1, j - 1
        elif state == UP:              # matrix Gb: gap in b
            aligned_a.append(a[i - 1])
            aligned_b.append("-")
            state = back_Gb[i][j]
            i -= 1
        else:                          # matrix Ga: gap in a
            aligned_a.append("-")
            aligned_b.append(b[j - 1])
            state = back_Ga[i][j]
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
        S, back = fill_linear(a, b, matrix, gap_open)
        best, cells = find_best_cells(S)
        alignments = [traceback_linear(a, b, matrix, S, back, cell) for cell in cells]
        return SmithWatermanResult(
            a, b, matrix, LINEAR, gap_open, gap_open, S, None, None, best, cells, alignments
        )

    S, Ga, Gb, back_S, back_Ga, back_Gb = fill_affine(a, b, matrix, gap_open, gap_extend)
    best, cells = find_best_cells(S)
    alignments = [
        traceback_affine(a, b, matrix, S, back_S, back_Ga, back_Gb, cell) for cell in cells
    ]
    return SmithWatermanResult(
        a, b, matrix, AFFINE, gap_open, gap_extend, S, Ga, Gb, best, cells, alignments
    )


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
