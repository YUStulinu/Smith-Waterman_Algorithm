"""Tests that drive the real tkinter window: buttons, examples, matrix view, errors.

The window is shown on screen while the tests run. Without a display (for
example on a headless Linux CI runner) the tests are skipped.
Run from the project folder with:  python -m unittest -v
"""

from __future__ import annotations

import gc
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gui
from smith_waterman import get_matrix, identity_matrix, smith_waterman

try:
    import tkinter as tk

    _probe = tk.Tk()
    _probe.destroy()
    del _probe
    DISPLAY = True
except Exception:  # tkinter missing, or no display to open a window on
    DISPLAY = False

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@unittest.skipUnless(DISPLAY, "no display available for tkinter")
class GuiTestCase(unittest.TestCase):
    def setUp(self):
        self.errors = mock.patch.object(gui.messagebox, "showerror").start()
        self.addCleanup(mock.patch.stopall)
        # Collect the destroyed window in the main thread: if the garbage
        # collector ran in the worker thread of the next test instead, every
        # tkinter Variable finaliser would wait for a main loop that is gone.
        self.app = gui.App()
        self.addCleanup(self.close_window)
        self.wait()

    def close_window(self):
        self.app.destroy()
        del self.app                  # unittest keeps the test case alive until the end
        gc.collect()

    def wait(self, timeout: float = 30.0) -> int:
        """Run the Tk event loop until the background alignment is done.

        Returns how many times the event loop ran meanwhile: the window
        stays responsive only if this keeps growing during a long alignment.
        """
        loops = 0
        deadline = time.perf_counter() + timeout
        self.app.update()
        while self.app.busy:
            self.assertLess(time.perf_counter(), deadline, "the alignment did not finish")
            self.app.update()
            loops += 1
            time.sleep(0.005)
        self.app.update()
        return loops

    def choose_example(self, index: int) -> None:
        self.app.example_box.current(index)
        self.app.example_box.event_generate("<<ComboboxSelected>>")
        self.wait()

    def type_sequences(self, a: str, b: str) -> None:
        self.app.set_sequence("a", a)
        self.app.set_sequence("b", b)
        self.app.seq_text["b"].focus_force()
        self.app.update()
        self.app.seq_text["b"].event_generate("<KeyRelease>", keysym="a")
        self.app.update()

    def press_align(self) -> None:
        self.app.align_button.invoke()
        self.wait()

    def click_cell(self, i: int, j: int) -> str:
        """Click cell (i, j) of the drawn matrix and return the explanation shown."""
        self.show_matrix_tab()
        self.app.canvas.event_generate("<Button-1>", x=int((j + 1.5) * gui.CELL), y=int((i + 1.5) * gui.CELL))
        self.app.update()
        return self.app.detail.get("1.0", "end").strip()

    def show_matrix_tab(self) -> None:
        matrix_tab = self.app.canvas.master.master
        matrix_tab.master.select(matrix_tab)
        self.app.update()

    def rectangles(self, fill: str | None = None) -> list[int]:
        items = self.app.canvas.find_all()
        rectangles = [item for item in items if self.app.canvas.type(item) == "rectangle"]
        if fill is None:
            return rectangles
        return [item for item in rectangles if self.app.canvas.itemcget(item, "fill") == fill]


class Examples(GuiTestCase):
    def test_first_example_is_aligned_at_start(self):
        self.assertEqual(self.app.result.best_score, 2)
        self.assertIn("Best score: 2", self.app.status_var.get())
        self.assertIn("Score:       2", self.app.report.get("1.0", "end"))

    def test_every_example_matches_the_library(self):
        for index, (label, a, b, matrix, gap, d, e) in enumerate(gui.EXAMPLES):
            with self.subTest(label):
                self.choose_example(index)
                expected = smith_waterman(a, b, get_matrix(matrix), int(d), int(e) if gap == "affine" else None)
                self.assertEqual(self.app.result.best_score, expected.best_score)
                self.assertEqual(self.app.result.gap_model, gap)
                self.assertEqual(self.app.matrix_var.get(), matrix)
                self.assertEqual(len(self.app.result.alignments), len(expected.alignments))
                self.assertIn(f"Best score: {expected.best_score}", self.app.status_var.get())

    def test_examples_ignored_while_busy(self):
        self.app.busy = True
        self.app.load_example(gui.EXAMPLES[4])
        self.app.align()
        self.app.busy = False
        self.assertEqual(self.app.example_box.get(), gui.EXAMPLES[0][0])


class MatrixView(GuiTestCase):
    def test_grid_is_drawn_with_the_traceback_path(self):
        rows, columns = len(self.app.result.a) + 1, len(self.app.result.b) + 1
        self.assertEqual(len(self.rectangles()), rows * columns)
        self.assertEqual(len(self.rectangles("#e8590c")), 1)                  # end of the alignment
        self.assertEqual(len(self.rectangles("#ffc078")), len(self.app.result.alignment.path) - 1)

    def test_click_explains_the_linear_recurrence(self):
        text = self.click_cell(5, 2)                   # S(5, 2) = 2 in the course example
        self.assertIn("S(5,2) = max{ 0,", text)
        self.assertIn("[diagonal: G aligned with G]", text)
        self.assertTrue(text.endswith("} = 2"))
        self.assertIn("initialisation", self.click_cell(0, 3))

    def test_nothing_to_draw_before_the_first_result(self):
        self.app.result = None
        self.app.draw_grid()
        self.assertEqual(self.app.canvas.find_all(), ())
        self.assertEqual(self.click_cell(1, 1), "")

    def test_click_outside_the_grid_does_nothing(self):
        self.assertIn("S(2,2)", self.click_cell(2, 2))
        before = self.app.detail.get("1.0", "end")
        self.app.canvas.event_generate("<Button-1>", x=5000, y=5000)
        self.app.update()
        self.assertEqual(self.app.detail.get("1.0", "end"), before)

    def test_affine_shows_three_matrices(self):
        self.choose_example(3)                         # linear vs affine example, affine gaps
        self.assertEqual(str(self.app.grid_buttons[1].cget("state")), "normal")
        self.assertIn("max{ S, Gb, Ga }", self.click_cell(4, 4))

        self.app.grid_buttons[1].invoke()              # Gb
        self.app.update()
        labels = {self.app.canvas.itemcget(item, "text") for item in self.app.canvas.find_all()
                  if self.app.canvas.type(item) == "text"}
        self.assertIn("-∞", labels)
        self.assertIn("[extend the gap]", self.click_cell(4, 4))
        self.assertIn("Gb(0,1) = -inf", self.click_cell(0, 1))

        self.app.grid_buttons[2].invoke()              # Ga
        self.app.update()
        self.assertIn("[gap in b followed by gap in a]", self.click_cell(4, 4))

    def test_linear_disables_gb_and_ga(self):
        self.choose_example(3)
        self.app.grid_buttons[2].invoke()
        self.choose_example(0)
        self.assertEqual(self.app.grid_var.get(), "S")
        for button in self.app.grid_buttons[1:]:
            self.assertEqual(str(button.cget("state")), "disabled")

    def test_large_matrix_is_not_drawn(self):
        self.type_sequences("ACGT" * 20, "ACGT" * 20)
        self.press_align()
        self.assertEqual(self.rectangles(), [])
        texts = [self.app.canvas.itemcget(item, "text") for item in self.app.canvas.find_all()]
        self.assertTrue(any("too large to draw" in text for text in texts))

    def test_no_alignment(self):
        self.type_sequences("AAAA", "TTTT")
        self.press_align()
        self.assertIsNone(self.app.result.alignment)
        self.assertEqual(self.rectangles("#e8590c"), [])
        self.assertIn("No local alignment", self.app.report.get("1.0", "end"))


class Parameters(GuiTestCase):
    def test_typing_a_protein_offers_protein_matrices(self):
        self.type_sequences("HEAGAWGHEE", "PAWHEAE")
        values = self.app.matrix_box.cget("values")
        self.assertIn("BLOSUM62", values)
        self.assertNotIn("IDENTITY", values)
        self.assertEqual(self.app.matrix_var.get(), "BLOSUM62")
        self.assertEqual(self.app.open_var.get(), "8")                     # suggested for BLOSUM
        self.assertEqual(str(self.app.match_entry.cget("state")), "disabled")

    def test_forced_type(self):
        self.app.type_var.set(gui.PROTEIN)
        self.app.refresh_matrix_choices()
        self.assertEqual(self.app.current_kind(), gui.PROTEIN)
        self.app.type_var.set("auto")
        self.app.set_sequence("a", "")
        self.app.set_sequence("b", "")
        self.assertEqual(self.app.current_kind(), gui.NUCLEOTIDE)

    def test_affine_suggests_both_penalties(self):
        self.app.gap_var.set("affine")
        self.app.on_gap_changed()
        self.assertEqual((self.app.open_var.get(), self.app.extend_var.get()), ("3", "1"))
        self.assertEqual(str(self.app.extend_entry.cget("state")), "normal")
        self.app.gap_var.set("linear")
        self.app.on_gap_changed()
        self.assertEqual(str(self.app.extend_entry.cget("state")), "disabled")

    def test_custom_identity_scores_and_decimal_comma(self):
        self.type_sequences("TGTTACGG", "GGTTGACTA")
        self.app.match_var.set("3")
        self.app.mismatch_var.set("-3")
        self.app.open_var.set("2,0")
        self.press_align()
        self.assertEqual(self.app.result.best_score, 13)

    def test_invalid_match_score_keeps_the_penalties(self):
        self.app.match_var.set("abc")
        self.app.open_var.set("5")
        self.app.suggest_penalties()
        self.assertEqual(self.app.open_var.get(), "5")


class InputErrors(GuiTestCase):
    def assert_error_shown(self, fragment: str) -> None:
        self.press_align()
        self.errors.assert_called()
        self.assertIn(fragment, self.errors.call_args.args[1])
        self.assertEqual(str(self.app.align_button.cget("state")), "normal")

    def test_invalid_symbol(self):
        self.type_sequences("ACGTX", "ACGT")
        self.app.type_var.set(gui.NUCLEOTIDE)
        self.app.refresh_matrix_choices()
        self.assert_error_shown("cannot be scored")

    def test_invalid_number(self):
        self.app.open_var.set("two")
        self.assert_error_shown("not a valid number for d")

    def test_error_raised_in_the_background_thread(self):
        self.app.gap_var.set("affine")
        self.app.open_var.set("1")
        self.app.extend_var.set("2")
        self.assert_error_shown("must not exceed")
        self.assertEqual(self.app.status_var.get(), "")


class Files(GuiTestCase):
    def test_load_fasta(self):
        path = str(EXAMPLES / "hbb_human.fasta")
        with mock.patch.object(gui.filedialog, "askopenfilename", return_value=path):
            self.app.load_fasta("a")
        self.assertEqual(len(gui.clean_sequence(self.app.get_sequence("a"))), 147)
        self.assertIn("HBB_HUMAN", self.app.status_var.get())

    def test_cancelled_dialogs_change_nothing(self):
        before = self.app.get_sequence("a")
        with mock.patch.object(gui.filedialog, "askopenfilename", return_value=""):
            self.app.load_fasta("a")
            self.app.load_matrix()
        self.assertEqual(self.app.get_sequence("a"), before)
        self.assertIsNone(self.app.custom_matrix)

    def test_unreadable_files(self):
        with tempfile.TemporaryDirectory() as folder:
            empty = Path(folder) / "empty.fasta"
            empty.write_text("")
            with mock.patch.object(gui.filedialog, "askopenfilename", return_value=str(empty)):
                self.app.load_fasta("a")
                self.app.load_matrix()
        self.assertEqual(self.errors.call_count, 2)

    def test_load_matrix_file(self):
        text = "   A  C  G  T\nA  2 -1 -1 -1\nC -1  2 -1 -1\nG -1 -1  2 -1\nT -1 -1 -1  2\n"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "my_dna.txt"
            path.write_text(text)
            with mock.patch.object(gui.filedialog, "askopenfilename", return_value=str(path)):
                self.app.load_matrix()
        self.assertEqual(self.app.matrix_var.get(), "my_dna")
        self.assertIn("my_dna", self.app.matrix_box.cget("values"))
        self.assertIs(self.app.selected_matrix(), self.app.custom_matrix)
        self.press_align()
        self.assertEqual(self.app.result.matrix.name, "my_dna")
        self.assertEqual(self.app.result.best_score, 4)            # TG / TG with +2 per match


class Responsiveness(GuiTestCase):
    def test_window_keeps_running_during_a_long_alignment(self):
        self.type_sequences("ACGT" * 300, "TGCA" * 300)           # 1200 x 1200 cells
        self.app.align_button.invoke()
        self.assertTrue(self.app.busy)
        self.assertEqual(str(self.app.align_button.cget("state")), "disabled")
        self.assertIn("Aligning 1200 x 1200", self.app.status_var.get())
        loops = self.wait()
        self.assertGreater(loops, 5, "the event loop was blocked while aligning")
        expected = smith_waterman("ACGT" * 300, "TGCA" * 300, identity_matrix(1, -1), 2)
        self.assertEqual(self.app.result.best_score, expected.best_score)


if __name__ == "__main__":
    unittest.main(verbosity=2)
