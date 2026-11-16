"""Smith-Waterman local sequence alignment (linear and affine gap penalties)."""

from .core import AFFINE, LINEAR, Alignment, SmithWatermanResult, score_alignment, smith_waterman
from .matrices import (
    NUCLEOTIDE,
    PROTEIN,
    SubstitutionMatrix,
    available_matrices,
    get_matrix,
    identity_matrix,
    load_matrix_file,
    transition_transversion_matrix,
)
from .sequences import clean_sequence, detect_type, prepare_sequence, read_fasta

__all__ = [
    "AFFINE", "LINEAR", "NUCLEOTIDE", "PROTEIN",
    "Alignment", "SmithWatermanResult", "SubstitutionMatrix",
    "available_matrices", "clean_sequence", "detect_type", "get_matrix",
    "identity_matrix", "load_matrix_file", "prepare_sequence", "read_fasta",
    "score_alignment", "smith_waterman", "transition_transversion_matrix",
]
