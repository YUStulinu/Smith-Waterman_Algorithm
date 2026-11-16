"""Reading, cleaning and validating biological sequences."""

from __future__ import annotations

from pathlib import Path

from .matrices import NUCLEOTIDE, PROTEIN, SubstitutionMatrix

NUCLEOTIDE_SYMBOLS = frozenset("ACGTUN")


def clean_sequence(text: str) -> str:
    """Normalise raw input: drop whitespace, digits and gap marks, upper-case.

    This makes it possible to paste a sequence copied from GenBank / UniProt,
    where residues are grouped in blocks and prefixed with position numbers.
    """
    return "".join(ch for ch in text.upper() if ch.isalpha() or ch == "*")


def detect_type(*sequences: str) -> str:
    """Guess whether the sequences are nucleotides or amino acids.

    If every symbol is one of A, C, G, T, U, N the input is treated as a
    nucleotide sequence, otherwise as a protein.
    """
    symbols = set("".join(sequences))
    return NUCLEOTIDE if symbols <= NUCLEOTIDE_SYMBOLS else PROTEIN


def prepare_sequence(text: str, matrix: SubstitutionMatrix) -> str:
    """Clean a sequence and check that the matrix can score every symbol.

    RNA is accepted for nucleotide matrices: U is aligned as T.
    """
    sequence = clean_sequence(text)
    if not sequence:
        raise ValueError("the sequence is empty")
    if matrix.kind == NUCLEOTIDE and "U" not in matrix:
        sequence = sequence.replace("U", "T")
    unknown = sorted(set(sequence) - set(matrix.alphabet))
    if unknown:
        raise ValueError(
            f"symbol(s) {', '.join(unknown)} cannot be scored with matrix {matrix.name} "
            f"(alphabet: {''.join(matrix.alphabet)})"
        )
    return sequence


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read all records of a FASTA file as (header, sequence) pairs.

    A file without any ``>`` header line is read as one raw sequence.
    """
    path = Path(path)
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if header is not None or chunks:
                records.append((header or path.stem, clean_sequence("".join(chunks))))
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line)
    if header is not None or chunks:
        records.append((header or path.stem, clean_sequence("".join(chunks))))
    if not records:
        raise ValueError(f"{path.name}: no sequence found")
    return records


def type_label(kind: str) -> str:
    return {NUCLEOTIDE: "nucleotide (DNA/RNA)", PROTEIN: "protein (amino acids)"}[kind]
