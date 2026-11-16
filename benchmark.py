"""Measures the running time of the implementation on random sequences.

The algorithm fills (n+1) x (m+1) cells, so doubling both lengths should
multiply the time by about four. Run with:  python benchmark.py
"""

from __future__ import annotations

import random
import time

from smith_waterman import get_matrix, identity_matrix, smith_waterman

LENGTHS = [100, 200, 400, 800, 1600]


def random_sequence(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def measure(alphabet: str, matrix, gap_open, gap_extend, length: int, repeats: int = 3) -> float:
    rng = random.Random(length)
    best = float("inf")
    for _ in range(repeats):
        a = random_sequence(rng, alphabet, length)
        b = random_sequence(rng, alphabet, length)
        start = time.perf_counter()
        smith_waterman(a, b, matrix, gap_open, gap_extend)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    cases = [
        ("DNA, linear gaps", "ACGT", identity_matrix(1, -1), 2, None),
        ("DNA, affine gaps", "ACGT", identity_matrix(1, -1), 3, 1),
        ("Protein BLOSUM62, linear", "ARNDCQEGHILKMFPSTWYV", get_matrix("BLOSUM62"), 8, None),
        ("Protein BLOSUM62, affine", "ARNDCQEGHILKMFPSTWYV", get_matrix("BLOSUM62"), 10, 1),
    ]
    print(f"{'n = m':>8}" + "".join(f"{name:>28}" for name, *_ in cases))
    previous = [None] * len(cases)
    for length in LENGTHS:
        row = f"{length:>8}"
        for index, (_name, alphabet, matrix, gap_open, gap_extend) in enumerate(cases):
            seconds = measure(alphabet, matrix, gap_open, gap_extend, length)
            ratio = f"(x{seconds / previous[index]:.1f})" if previous[index] else ""
            row += f"{seconds:>18.3f} s {ratio:>7}"
            previous[index] = seconds
        print(row)


if __name__ == "__main__":
    main()
