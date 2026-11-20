"""Substitution (scoring) matrices for nucleotide and amino acid sequences.

A substitution matrix M gives the score M(x, y) of aligning symbol x with
symbol y. Gap penalties are *not* stored in the matrix; they are separate
parameters of the alignment algorithm (see ``core.py``).

Available matrices
------------------
Nucleotides:
    IDENTITY                 match / mismatch scores chosen by the user
    TRANSITION_TRANSVERSION  mismatches penalised by mutation type (course 4)
    NUC.4.4                  EDNAFULL matrix used by EMBOSS / NCBI (+5 / -4)
Amino acids:
    BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90
    PAM30, PAM70, PAM250

Any other matrix can be loaded from a text file in the NCBI format with
``load_matrix_file``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

NUCLEOTIDE = "nucleotide"
PROTEIN = "protein"

PURINES = frozenset("AG")
PYRIMIDINES = frozenset("CT")

NUCLEOTIDE_MATRICES = ("IDENTITY", "TRANSITION_TRANSVERSION", "NUC.4.4")
PROTEIN_MATRICES = (
    "BLOSUM62", "BLOSUM45", "BLOSUM50", "BLOSUM80", "BLOSUM90",
    "PAM250", "PAM30", "PAM70",
)

DESCRIPTIONS = {
    "IDENTITY": "same score for every match, same penalty for every mismatch",
    "TRANSITION_TRANSVERSION": "transitions (A<->G, C<->T) penalised less than transversions",
    "NUC.4.4": "EDNAFULL: +5 match, -4 mismatch, supports IUPAC ambiguity codes",
    "BLOSUM45": "distantly related proteins",
    "BLOSUM50": "moderately distant proteins",
    "BLOSUM62": "general purpose default (used by BLASTP)",
    "BLOSUM80": "closely related proteins",
    "BLOSUM90": "very closely related proteins",
    "PAM30": "very closely related proteins, short alignments",
    "PAM70": "closely related proteins",
    "PAM250": "distantly related proteins",
}


@dataclass(frozen=True)
class SubstitutionMatrix:
    """A symmetric table of substitution scores over a fixed alphabet."""

    name: str
    kind: str                       # NUCLEOTIDE or PROTEIN
    alphabet: tuple[str, ...]
    scores: dict[tuple[str, str], int | float]

    def score(self, x: str, y: str) -> int | float:
        """Return M(x, y)."""
        return self.scores[(x, y)]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.alphabet

    def as_table(self) -> str:
        """Render the matrix as text (used by ``--show-substitution``)."""
        width = max(len(_fmt(v)) for v in self.scores.values()) + 1
        lines = [" " + "".join(s.rjust(width) for s in self.alphabet)]
        for x in self.alphabet:
            row = "".join(_fmt(self.scores[(x, y)]).rjust(width) for y in self.alphabet)
            lines.append(x + row)
        return "\n".join(lines)


def _fmt(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def identity_matrix(match: int | float = 1, mismatch: int | float = -1) -> SubstitutionMatrix:
    """Nucleotide matrix with one match score and one mismatch score."""
    if not (math.isfinite(match) and math.isfinite(mismatch)):
        raise ValueError("the match and mismatch scores must be finite numbers")
    if match <= 0:
        raise ValueError("the match score must be positive")
    if mismatch >= 0:
        raise ValueError("the mismatch score must be negative")
    alphabet = ("A", "C", "G", "T")
    scores = {(x, y): (match if x == y else mismatch) for x in alphabet for y in alphabet}
    return SubstitutionMatrix(f"IDENTITY({_fmt(match)}/{_fmt(mismatch)})", NUCLEOTIDE, alphabet, scores)


def transition_transversion_matrix(
    match: int | float = 1,
    transition: int | float = -1,
    transversion: int | float = -5,
) -> SubstitutionMatrix:
    """Nucleotide matrix that distinguishes the two kinds of point mutations.

    A transition replaces a purine with a purine (A<->G) or a pyrimidine with a
    pyrimidine (C<->T); it is biologically more frequent, so it is penalised
    less than a transversion (purine <-> pyrimidine).
    """
    alphabet = ("A", "C", "G", "T")
    scores = {}
    for x in alphabet:
        for y in alphabet:
            if x == y:
                scores[(x, y)] = match
            elif {x, y} <= PURINES or {x, y} <= PYRIMIDINES:
                scores[(x, y)] = transition
            else:
                scores[(x, y)] = transversion
    return SubstitutionMatrix("TRANSITION_TRANSVERSION", NUCLEOTIDE, alphabet, scores)


def load_matrix_file(path: str | Path, name: str | None = None, kind: str | None = None) -> SubstitutionMatrix:
    """Load a matrix written in the NCBI text format.

    Lines starting with ``#`` are comments. The first remaining line lists the
    column symbols; every following line starts with a row symbol followed by
    one score per column.
    """
    path = Path(path)
    # utf-8-sig also strips the BOM that some Windows editors add
    rows = [
        line.split()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) < 2:
        raise ValueError(f"{path.name}: not a substitution matrix file")

    columns = [c.upper() for c in rows[0]]
    scores: dict[tuple[str, str], int | float] = {}
    alphabet: list[str] = []
    for row in rows[1:]:
        symbol, values = row[0].upper(), row[1:]
        if len(values) != len(columns):
            raise ValueError(f"{path.name}: row '{symbol}' has {len(values)} scores, expected {len(columns)}")
        alphabet.append(symbol)
        for column, value in zip(columns, values):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{path.name}: score '{value}' for ({symbol}, {column}) is not a finite number")
            scores[(symbol, column)] = int(number) if number.is_integer() else number

    if set(alphabet) != set(columns):
        raise ValueError(f"{path.name}: row symbols and column symbols differ")
    for x in alphabet:
        for y in alphabet:
            if scores[(x, y)] != scores[(y, x)]:
                raise ValueError(f"{path.name}: matrix is not symmetric at ({x}, {y})")

    if kind is None:
        kind = PROTEIN if len(alphabet) > 16 else NUCLEOTIDE
    return SubstitutionMatrix(name or path.stem, kind, tuple(alphabet), scores)


def available_matrices(kind: str) -> tuple[str, ...]:
    """Names of the built-in matrices suitable for a sequence type."""
    if kind == NUCLEOTIDE:
        return NUCLEOTIDE_MATRICES
    if kind == PROTEIN:
        return PROTEIN_MATRICES
    raise ValueError(f"unknown sequence type: {kind}")


def default_matrix_name(kind: str) -> str:
    return available_matrices(kind)[0]


def get_matrix(
    name: str,
    match: int | float = 1,
    mismatch: int | float = -1,
) -> SubstitutionMatrix:
    """Return a built-in matrix by name (case-insensitive).

    ``match`` and ``mismatch`` are used only by the IDENTITY matrix.
    """
    key = name.strip().upper()
    if key == "IDENTITY":
        return identity_matrix(match, mismatch)
    if key == "TRANSITION_TRANSVERSION":
        return transition_transversion_matrix()
    if key in NUCLEOTIDE_MATRICES:
        return load_matrix_file(DATA_DIR / f"{key}.txt", key, NUCLEOTIDE)
    if key in PROTEIN_MATRICES:
        return load_matrix_file(DATA_DIR / f"{key}.txt", key, PROTEIN)
    known = ", ".join(NUCLEOTIDE_MATRICES + PROTEIN_MATRICES)
    raise ValueError(f"unknown substitution matrix '{name}'. Available: {known}")
