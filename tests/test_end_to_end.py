"""End-to-end tests: the programs are started as separate processes, as a user would.

Run from the project folder with:  python -m unittest -v
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark
from smith_waterman import identity_matrix

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def run(*arguments: str, stdin: str = "", timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, input=stdin, capture_output=True,
        text=True, encoding="utf-8", env=ENV, timeout=timeout, check=False,
    )


class Programs(unittest.TestCase):
    def test_cli_course_example(self):
        process = run("main.py", "--seq1", "ATCTGA", "--seq2", "TGCA", "--gap-open", "2", "--show-matrix")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Score:       2", process.stdout)
        self.assertIn("a 4 TG 5", process.stdout)
        self.assertIn("b 1 TG 2", process.stdout)

    def test_cli_hemoglobins(self):
        process = run("main.py", "--fasta1", "examples/hbb_human.fasta", "--fasta2", "examples/hba_human.fasta",
                      "--matrix", "BLOSUM62", "--gap", "affine", "--gap-open", "10", "--gap-extend", "1")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Score:       291", process.stdout)

    def test_cli_error_exit_code(self):
        process = run("main.py", "--seq1", "ACGT", "--seq2", "ACGT", "--matrix", "BLOSUM62")
        self.assertEqual(process.returncode, 1)
        self.assertIn("Error:", process.stderr)
        self.assertEqual(process.stdout, "")

    def test_cli_bad_option_exit_code(self):
        process = run("main.py", "--seq1", "ACGT", "--gap", "cubic")
        self.assertEqual(process.returncode, 2)
        self.assertIn("invalid choice", process.stderr)

    def test_interactive_mode_through_stdin(self):
        process = run("main.py", stdin="TGTTACGG\nGGTTGACTA\nIDENTITY\n3\n-3\nlinear\n2\ny\n")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Score:       13", process.stdout)
        self.assertIn("Score matrix S", process.stdout)

    def test_demo_runs_every_example(self):
        process = run("demo.py")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.count("Smith-Waterman local alignment"), 10)
        for score in ("Score:       2", "Score:       13", "Score:       28", "Score:       291"):
            self.assertIn(score, process.stdout)

    def test_gui_starts_and_stays_open(self):
        process = subprocess.Popen([sys.executable, "gui.py"], cwd=ROOT, env=ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            _, error = process.communicate(timeout=3)      # returns as soon as gui.py exits
            if "display" in error.lower():
                self.skipTest("no display available for tkinter")
            self.fail(f"gui.py exited with code {process.returncode}:\n{error}")
        except subprocess.TimeoutExpired:
            pass                                           # still running after 3 s: it stayed open
        finally:
            process.kill()
            process.communicate()


class Benchmark(unittest.TestCase):
    def test_random_sequence(self):
        sequence = benchmark.random_sequence(random.Random(1), "ACGT", 50)
        self.assertEqual(len(sequence), 50)
        self.assertLessEqual(set(sequence), set("ACGT"))

    def test_table(self):
        out = io.StringIO()
        with mock.patch.object(benchmark, "LENGTHS", [40, 80]), contextlib.redirect_stdout(out):
            benchmark.main()
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("Protein BLOSUM62, affine", lines[0])
        self.assertEqual(lines[2].count("(x"), 4)          # one growth ratio per case

    def test_measure_returns_the_best_time(self):
        seconds = benchmark.measure("ACGT", identity_matrix(), 2, None, 50, repeats=2)
        self.assertGreater(seconds, 0)
        self.assertLess(seconds, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
