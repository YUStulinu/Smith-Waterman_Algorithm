"""Graphical interface for the Smith-Waterman local alignment.

Run with ``python gui.py``. Only the Python standard library (tkinter) is used.

The window has two result tabs:
  * Alignment    - the report (score, statistics, aligned sequences)
  * DP matrix    - the dynamic programming matrix with the traceback path
                   highlighted; click a cell to see how its value was computed
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from smith_waterman.cli import default_gap_penalties
from smith_waterman.core import AFFINE, NEG_INF, SmithWatermanResult, smith_waterman
from smith_waterman.formatting import format_number, format_result
from smith_waterman.matrices import (
    NUCLEOTIDE,
    PROTEIN,
    available_matrices,
    get_matrix,
    load_matrix_file,
)
from smith_waterman.sequences import clean_sequence, detect_type, prepare_sequence, read_fasta

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
MAX_DRAWN_CELLS = 70 * 70
MAX_LISTED_ALIGNMENTS = 100     # more are only counted: listing 360000 froze the window for seconds
CELL = 38

# (label, sequence a, sequence b, matrix, gap model, d, e)
EXAMPLES = [
    ("Course example (DNA, linear)", "ATCTGA", "TGCA", "IDENTITY", "linear", "2", ""),
    ("Lab example (DNA, linear)", "TGTTACGG", "GGTTGACTA", "IDENTITY", "linear", "1", ""),
    ("Transitions vs transversions", "GATTACAGGT", "CGACTGCAGA", "TRANSITION_TRANSVERSION", "linear", "3", ""),
    ("Linear vs affine: try both", "ACGTTTGACCATGCA", "ACGTTTCATGCA", "IDENTITY", "affine", "3", "1"),
    ("Protein, BLOSUM50 (Durbin et al.)", "HEAGAWGHEE", "PAWHEAE", "BLOSUM50", "linear", "8", ""),
    ("Protein, BLOSUM62, affine", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "KTAYIAKQRSFVKSHFQLEERLG", "BLOSUM62", "affine", "10", "1"),
]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Smith-Waterman local alignment")
        self.geometry("1100x800")
        self.minsize(900, 600)

        self.result: SmithWatermanResult | None = None
        self.busy = False                   # True while an alignment runs in the background
        self.custom_matrix = None           # matrix loaded from a file, if any

        self.type_var = tk.StringVar(value="auto")
        self.matrix_var = tk.StringVar(value="IDENTITY")
        self.match_var = tk.StringVar(value="1")
        self.mismatch_var = tk.StringVar(value="-1")
        self.gap_var = tk.StringVar(value="linear")
        self.open_var = tk.StringVar(value="2")
        self.extend_var = tk.StringVar(value="1")
        self.grid_var = tk.StringVar(value="S")
        self.status_var = tk.StringVar(value="Enter two sequences and press Align.")

        self._build_inputs()
        self._build_parameters()
        # The status bar is packed before the results so that it keeps its
        # place at the bottom when the results area expands.
        ttk.Label(self, textvariable=self.status_var, anchor="w", relief="sunken").pack(fill="x", side="bottom")
        self._build_results()

        self.load_example(EXAMPLES[0])

    # ------------------------------------------------------------------ layout

    def _build_inputs(self) -> None:
        frame = ttk.LabelFrame(self, text="Sequences")
        frame.pack(fill="x", padx=8, pady=(8, 4))
        frame.columnconfigure(1, weight=1)

        self.seq_text: dict[str, tk.Text] = {}
        for row, key in enumerate(("a", "b")):
            ttk.Label(frame, text=f"Sequence {key}:").grid(row=row, column=0, sticky="nw", padx=6, pady=4)
            text = tk.Text(frame, height=2, wrap="char", font=("Consolas", 11))
            text.grid(row=row, column=1, sticky="ew", pady=4)
            text.bind("<KeyRelease>", lambda _event: self.on_sequences_changed())
            self.seq_text[key] = text
            ttk.Button(frame, text="Load FASTA...", command=lambda k=key: self.load_fasta(k)).grid(
                row=row, column=2, padx=6, pady=4, sticky="n")

        bar = ttk.Frame(frame)
        bar.grid(row=2, column=1, sticky="w", pady=(0, 6))
        ttk.Label(bar, text="Example:").pack(side="left")
        self.example_box = ttk.Combobox(bar, state="readonly", width=38, values=[e[0] for e in EXAMPLES])
        self.example_box.pack(side="left", padx=6)
        self.example_box.bind("<<ComboboxSelected>>",
                              lambda _event: self.load_example(EXAMPLES[self.example_box.current()]))

    def _build_parameters(self) -> None:
        frame = ttk.LabelFrame(self, text="Parameters")
        frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(frame, text="Type:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        for column, (label, value) in enumerate((("Auto", "auto"), ("DNA/RNA", NUCLEOTIDE), ("Protein", PROTEIN)), 1):
            ttk.Radiobutton(frame, text=label, value=value, variable=self.type_var,
                            command=self.refresh_matrix_choices).grid(row=0, column=column, sticky="w")

        ttk.Label(frame, text="Substitution matrix:").grid(row=0, column=4, padx=(24, 6), sticky="w")
        self.matrix_box = ttk.Combobox(frame, state="readonly", width=26, textvariable=self.matrix_var)
        self.matrix_box.grid(row=0, column=5, sticky="w")
        self.matrix_box.bind("<<ComboboxSelected>>", lambda _event: self.on_matrix_changed())
        ttk.Button(frame, text="From file...", command=self.load_matrix).grid(row=0, column=6, padx=6)

        ttk.Label(frame, text="Match:").grid(row=0, column=7, padx=(16, 4))
        self.match_entry = ttk.Entry(frame, width=5, textvariable=self.match_var)
        self.match_entry.grid(row=0, column=8)
        ttk.Label(frame, text="Mismatch:").grid(row=0, column=9, padx=(10, 4))
        self.mismatch_entry = ttk.Entry(frame, width=5, textvariable=self.mismatch_var)
        self.mismatch_entry.grid(row=0, column=10)

        ttk.Label(frame, text="Gap penalty:").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        ttk.Radiobutton(frame, text="Linear", value="linear", variable=self.gap_var,
                        command=self.on_gap_changed).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(frame, text="Affine", value="affine", variable=self.gap_var,
                        command=self.on_gap_changed).grid(row=1, column=2, sticky="w")
        ttk.Label(frame, text="d (gap / gap opening):").grid(row=1, column=4, padx=(24, 6), sticky="w")
        ttk.Entry(frame, width=6, textvariable=self.open_var).grid(row=1, column=5, sticky="w")
        ttk.Label(frame, text="e (extension):").grid(row=1, column=7, columnspan=2, padx=(16, 4), sticky="e")
        self.extend_entry = ttk.Entry(frame, width=5, textvariable=self.extend_var)
        self.extend_entry.grid(row=1, column=10)

        self.align_button = ttk.Button(frame, text="Align", command=self.align)
        self.align_button.grid(row=0, column=11, rowspan=2, padx=16, ipadx=12, ipady=6)

    def _build_results(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        report_tab = ttk.Frame(notebook)
        notebook.add(report_tab, text="Alignment")
        self.report = tk.Text(report_tab, font=("Consolas", 11), wrap="none", state="disabled")
        scroll_y = ttk.Scrollbar(report_tab, orient="vertical", command=self.report.yview)
        scroll_x = ttk.Scrollbar(report_tab, orient="horizontal", command=self.report.xview)
        self.report.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.report.pack(fill="both", expand=True)

        matrix_tab = ttk.Frame(notebook)
        notebook.add(matrix_tab, text="DP matrix")
        top = ttk.Frame(matrix_tab)
        top.pack(fill="x")
        ttk.Label(top, text="Show:").pack(side="left", padx=6)
        self.grid_buttons = [
            ttk.Radiobutton(top, text=name, value=name, variable=self.grid_var, command=self.draw_grid)
            for name in ("S", "Gb", "Ga")
        ]
        for button in self.grid_buttons:
            button.pack(side="left")
        ttk.Label(top, text="   Click a cell to see how it was computed.", foreground="#555").pack(side="left")

        self.detail = tk.Text(matrix_tab, height=6, font=("Consolas", 10), state="disabled", background="#f7f7f7")
        self.detail.pack(fill="x", side="bottom")

        holder = ttk.Frame(matrix_tab)
        holder.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(holder, background="white", highlightthickness=0)
        scroll_y = ttk.Scrollbar(holder, orient="vertical", command=self.canvas.yview)
        scroll_x = ttk.Scrollbar(holder, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_cell_click)

    # ----------------------------------------------------------------- helpers

    def get_sequence(self, key: str) -> str:
        return self.seq_text[key].get("1.0", "end")

    def set_sequence(self, key: str, value: str) -> None:
        self.seq_text[key].delete("1.0", "end")
        self.seq_text[key].insert("1.0", value)

    def current_kind(self) -> str:
        if self.type_var.get() != "auto":
            return self.type_var.get()
        a, b = clean_sequence(self.get_sequence("a")), clean_sequence(self.get_sequence("b"))
        return detect_type(a, b) if (a or b) else NUCLEOTIDE

    def on_sequences_changed(self) -> None:
        """Follow an edit of the sequences: drop a stale example label, refresh the matrices."""
        index = self.example_box.current()
        typed = (clean_sequence(self.get_sequence("a")), clean_sequence(self.get_sequence("b")))
        if index >= 0 and typed != EXAMPLES[index][1:3]:
            self.example_box.set("")
        self.refresh_matrix_choices()

    def refresh_matrix_choices(self) -> None:
        """Offer only the matrices that fit the current sequence type."""
        names = list(available_matrices(self.current_kind()))
        if self.custom_matrix is not None and self.custom_matrix.kind == self.current_kind():
            names.append(self.custom_matrix.name)
        self.matrix_box.configure(values=names)
        if self.matrix_var.get() not in names:
            self.matrix_var.set(names[0])
            self.suggest_penalties()
        self.on_matrix_changed(suggest=False)

    def on_matrix_changed(self, suggest: bool = True) -> None:
        state = "normal" if self.matrix_var.get() == "IDENTITY" else "disabled"
        self.match_entry.configure(state=state)
        self.mismatch_entry.configure(state=state)
        if suggest:
            self.suggest_penalties()

    def on_gap_changed(self) -> None:
        self.extend_entry.configure(state="normal" if self.gap_var.get() == "affine" else "disabled")
        self.suggest_penalties()

    def suggest_penalties(self) -> None:
        """Fill d and e with defaults that suit the scale of the chosen matrix."""
        try:
            matrix = self.selected_matrix()
        except ValueError:
            return
        gap_open, gap_extend = default_gap_penalties(matrix, self.gap_var.get())
        self.open_var.set(str(gap_open))
        if gap_extend is not None:
            self.extend_var.set(str(gap_extend))

    def selected_matrix(self):
        name = self.matrix_var.get()
        if self.custom_matrix is not None and name == self.custom_matrix.name:
            return self.custom_matrix
        return get_matrix(name, self.parse_number(self.match_var.get(), "match"),
                          self.parse_number(self.mismatch_var.get(), "mismatch"))

    @staticmethod
    def parse_number(text: str, label: str) -> int | float:
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            raise ValueError(f"'{text}' is not a valid number for {label}") from None
        return int(value) if value.is_integer() else value

    # ----------------------------------------------------------------- actions

    def load_example(self, example: tuple) -> None:
        if self.busy:
            return
        label, a, b, matrix, gap, gap_open, gap_extend = example
        self.example_box.set(label)
        self.custom_matrix = None
        self.type_var.set("auto")
        self.set_sequence("a", a)
        self.set_sequence("b", b)
        self.refresh_matrix_choices()
        self.matrix_var.set(matrix)
        self.match_var.set("1")
        self.mismatch_var.set("-1")
        self.gap_var.set(gap)
        self.on_matrix_changed(suggest=False)
        self.extend_entry.configure(state="normal" if gap == "affine" else "disabled")
        self.open_var.set(gap_open)
        if gap_extend:
            self.extend_var.set(gap_extend)
        self.align()

    def load_fasta(self, key: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Sequence {key}", initialdir=EXAMPLES_DIR,
            filetypes=[("FASTA / text", "*.fasta *.fa *.faa *.fna *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            records = read_fasta(path)
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot read file", str(error))
            return
        header, sequence = records[0]
        self.set_sequence(key, sequence)
        self.on_sequences_changed()
        self.status_var.set(f"Loaded sequence {key}: {header} ({len(sequence)} symbols)")

    def load_matrix(self) -> None:
        path = filedialog.askopenfilename(title="Substitution matrix (NCBI format)")
        if not path:
            return
        try:
            self.custom_matrix = load_matrix_file(path, kind=self.current_kind())
        except (ValueError, OSError) as error:
            messagebox.showerror("Cannot read matrix", str(error))
            return
        self.refresh_matrix_choices()
        self.matrix_var.set(self.custom_matrix.name)
        self.on_matrix_changed()

    def align(self) -> None:
        """Read the inputs and start the alignment in a background thread.

        Long sequences take a few seconds; computing outside the Tk event
        loop keeps the window responsive. Tk widgets are only touched from
        the main thread: finish_align polls the worker with after().
        """
        if self.busy:
            return
        try:
            matrix = self.selected_matrix()
            a = prepare_sequence(self.get_sequence("a"), matrix)
            b = prepare_sequence(self.get_sequence("b"), matrix)
            gap_open = self.parse_number(self.open_var.get(), "d")
            gap_extend = None
            if self.gap_var.get() == "affine":
                gap_extend = self.parse_number(self.extend_var.get(), "e")
        except ValueError as error:
            messagebox.showerror("Invalid input", str(error))
            return

        outcome: dict = {}

        def work() -> None:
            try:
                outcome["result"] = smith_waterman(a, b, matrix, gap_open, gap_extend)
            except ValueError as error:
                outcome["error"] = error

        self.busy = True
        self.align_button.configure(state="disabled")
        self.example_box.configure(state="disabled")    # a pick now would only change the label
        self.status_var.set(f"Aligning {len(a)} x {len(b)} symbols...")
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.after(20, self.finish_align, worker, outcome)

    def finish_align(self, worker: threading.Thread, outcome: dict) -> None:
        if worker.is_alive():
            self.after(20, self.finish_align, worker, outcome)
            return
        self.busy = False
        self.align_button.configure(state="normal")
        self.example_box.configure(state="readonly")
        if "error" in outcome:
            self.status_var.set("")
            messagebox.showerror("Invalid input", str(outcome["error"]))
            return

        self.result = outcome["result"]
        show_all = len(self.result.alignments) <= MAX_LISTED_ALIGNMENTS
        text = format_result(self.result, "a", "b", show_matrix=False, show_all=show_all)
        self.set_text(self.report, text)

        affine = self.result.gap_model == AFFINE
        if not affine:
            self.grid_var.set("S")
        for button in self.grid_buttons[1:]:
            button.configure(state="normal" if affine else "disabled")
        self.set_text(self.detail, "")
        self.draw_grid()
        count = len(self.result.alignments)
        self.status_var.set(
            f"Best score: {format_number(self.result.best_score)}   |   optimal alignments found: {count}   |   "
            f"matrix size: {len(self.result.a) + 1} x {len(self.result.b) + 1}")

    @staticmethod
    def set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # ------------------------------------------------------------- matrix view

    def shown_grid(self):
        result = self.result
        return {"S": result.S, "Gb": result.Gb, "Ga": result.Ga}[self.grid_var.get()]

    def draw_grid(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        result = self.result
        if result is None:
            return
        rows, columns = len(result.a) + 1, len(result.b) + 1
        if rows * columns > MAX_DRAWN_CELLS:
            canvas.create_text(20, 20, anchor="nw", font=("Segoe UI", 11),
                               text=f"The matrix has {rows} x {columns} cells and is too large to draw.\n"
                                    "The alignment itself is shown in the Alignment tab.")
            canvas.configure(scrollregion=(0, 0, 600, 100))
            return

        grid = self.shown_grid()
        path = set(result.alignment.path) if result.alignment else set()
        end = result.alignment.path[-1] if result.alignment else None
        finite = [v for row in grid for v in row if v != NEG_INF]
        top = max(max(finite), 1)

        header_font = ("Consolas", 11, "bold")
        for j in range(columns):
            if j > 0:
                canvas.create_text((j + 1.5) * CELL, CELL / 2, text=result.b[j - 1], font=header_font)
        for i in range(rows):
            if i > 0:
                canvas.create_text(CELL / 2, (i + 1.5) * CELL, text=result.a[i - 1], font=header_font)
            for j in range(columns):
                value = grid[i][j]
                x, y = (j + 1) * CELL, (i + 1) * CELL
                if (i, j) == end:
                    fill = "#e8590c"
                elif (i, j) in path:
                    fill = "#ffc078"
                elif value == NEG_INF or value <= 0:
                    fill = "#ffffff"
                else:
                    shade = int(255 - 110 * (value / top))
                    fill = f"#{shade:02x}{min(255, shade + 20):02x}ff"
                canvas.create_rectangle(x, y, x + CELL, y + CELL, fill=fill, outline="#cccccc")
                label = "-∞" if value == NEG_INF else format_number(value)
                canvas.create_text(x + CELL / 2, y + CELL / 2, text=label, font=("Consolas", 10),
                                   fill="white" if (i, j) == end else "black")
        canvas.configure(scrollregion=(0, 0, (columns + 1) * CELL + 10, (rows + 1) * CELL + 10))

    def on_cell_click(self, event: tk.Event) -> None:
        result = self.result
        if result is None:
            return
        j = int(self.canvas.canvasx(event.x) // CELL) - 1
        i = int(self.canvas.canvasy(event.y) // CELL) - 1
        if not (0 <= i <= len(result.a) and 0 <= j <= len(result.b)):
            return
        self.set_text(self.detail, self.explain_cell(i, j))

    def explain_cell(self, i: int, j: int) -> str:
        """Spell out the recurrence for cell (i, j) of the displayed matrix."""
        result = self.result
        name = self.grid_var.get()
        fmt = format_number
        d, e = fmt(result.gap_open), fmt(result.gap_extend)
        value = fmt(self.shown_grid()[i][j])
        if i == 0 or j == 0:
            rule = "0" if name == "S" else "-inf"
            return f"{name}({i},{j}) = {rule}   (initialisation of the first row / column)"

        x, y = result.a[i - 1], result.b[j - 1]
        S, Ga, Gb = result.S, result.Ga, result.Gb
        sub = result.matrix.score(x, y)

        if result.gap_model != AFFINE:
            return "\n".join([
                f"S({i},{j}) = max{{ 0,",
                f"    S({i - 1},{j - 1}) + M({x},{y}) = {fmt(S[i - 1][j - 1])} + ({fmt(sub)}) = {fmt(S[i - 1][j - 1] + sub)}   [diagonal: {x} aligned with {y}]",
                f"    S({i - 1},{j}) - d = {fmt(S[i - 1][j])} - {d} = {fmt(S[i - 1][j] - result.gap_open)}   [up: {x} aligned with a gap]",
                f"    S({i},{j - 1}) - d = {fmt(S[i][j - 1])} - {d} = {fmt(S[i][j - 1] - result.gap_open)}   [left: {y} aligned with a gap]",
                f"}} = {value}",
            ])
        if name == "S":
            return "\n".join([
                f"S({i},{j}) = max{{ 0, max{{ S, Gb, Ga }}({i - 1},{j - 1}) + M({x},{y}) }}",
                f"    S({i - 1},{j - 1})  = {fmt(S[i - 1][j - 1])}",
                f"    Gb({i - 1},{j - 1}) = {fmt(Gb[i - 1][j - 1])}",
                f"    Ga({i - 1},{j - 1}) = {fmt(Ga[i - 1][j - 1])}",
                f"    M({x},{y}) = {fmt(sub)}",
                f"= {value}",
            ])
        if name == "Gb":
            return "\n".join([
                f"Gb({i},{j}) = max{{   [{x} aligned with a gap in b]",
                f"    S({i - 1},{j}) - d  = {fmt(S[i - 1][j])} - {d} = {fmt(S[i - 1][j] - result.gap_open)}   [open a gap]",
                f"    Gb({i - 1},{j}) - e = {fmt(Gb[i - 1][j])} - {e} = {fmt(Gb[i - 1][j] - result.gap_extend)}   [extend the gap]",
                f"    Ga({i - 1},{j}) - d = {fmt(Ga[i - 1][j])} - {d} = {fmt(Ga[i - 1][j] - result.gap_open)}   [gap in a followed by gap in b]",
                f"}} = {value}",
            ])
        return "\n".join([
            f"Ga({i},{j}) = max{{   [{y} aligned with a gap in a]",
            f"    S({i},{j - 1}) - d  = {fmt(S[i][j - 1])} - {d} = {fmt(S[i][j - 1] - result.gap_open)}   [open a gap]",
            f"    Ga({i},{j - 1}) - e = {fmt(Ga[i][j - 1])} - {e} = {fmt(Ga[i][j - 1] - result.gap_extend)}   [extend the gap]",
            f"    Gb({i},{j - 1}) - d = {fmt(Gb[i][j - 1])} - {d} = {fmt(Gb[i][j - 1] - result.gap_open)}   [gap in b followed by gap in a]",
            f"}} = {value}",
        ])


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
