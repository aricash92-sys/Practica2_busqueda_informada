"""
app.py – Interfaz gráfica para la Práctica 2.
Ejercicio 1: 8-puzzle / 15-puzzle con A* y 3 heurísticas.
Ejercicio 2: Sudoku con A* y Recocido Simulado.

Requiere sólo la stdlib de Python 3 (tkinter incluido).
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, font as tkfont

# ── importar lógica del backend ──────────────────────────────────────────────
from puzzle import (
    a_star,
    misplaced_tiles,
    manhattan_distance,
    custom_heuristic,
    GOAL_8, GOAL_15,
    PUZZLES_8, PUZZLES_15,
    generate_random_puzzle,
)
from sudoku import (
    a_star_sudoku,
    simulated_annealing,
    EASY, MEDIUM, HARD,
    LEVELS,
    generate_random_sudoku,
)

# ── Paleta de colores ─────────────────────────────────────────────────────────
BG        = "#0f1117"
BG2       = "#1a1d27"
BG3       = "#22263a"
ACCENT    = "#6c63ff"
ACCENT2   = "#a78bfa"
SUCCESS   = "#22c55e"
ERROR     = "#ef4444"
WARNING   = "#f59e0b"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#64748b"
BORDER    = "#334155"
TILE_CLR  = {
    "empty":    "#1e293b",
    "default":  "#334155",
    "hover":    "#475569",
    "selected": "#6c63ff",
}

# Heurísticas disponibles
HEURISTICS = [
    ("Fichas fuera de lugar", misplaced_tiles),
    ("Distancia Manhattan",   manhattan_distance),
    ("Heurística personalizada", custom_heuristic),
]

PUZZLE_LABELS  = ["Fácil", "Medio", "Difícil"]
SUDOKU_LABELS  = ["Fácil (20 vacías)", "Intermedio (35 vacías)", "Difícil (45 vacías)"]
SUDOKU_PUZZLES = [EASY, MEDIUM, HARD]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: tabla con scrollbar
# ─────────────────────────────────────────────────────────────────────────────

class StyledTable(tk.Frame):
    """Tabla simple usando tk.Canvas + tk.Frame interior (sin ttk.Treeview)."""

    HEADER_BG = BG3
    ROW_BG    = [BG2, "#1e2235"]
    COL_W     = 155

    def __init__(self, master, headers, col_widths=None, **kw):
        super().__init__(master, bg=BG2, **kw)
        self.headers    = headers
        self.col_widths = col_widths or [self.COL_W] * len(headers)
        self._build()

    def _build(self):
        # Canvas + scrollbar
        self.canvas = tk.Canvas(self, bg=BG2, highlightthickness=0)
        vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=BG2)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e:
            self.canvas.itemconfig(self._win, width=e.width))
        self._draw_header()
        self._row_idx = 0

    def _draw_header(self):
        for c, (h, w) in enumerate(zip(self.headers, self.col_widths)):
            tk.Label(self.inner, text=h, bg=self.HEADER_BG, fg=ACCENT2,
                     font=("Segoe UI", 9, "bold"), width=w // 9,
                     relief="flat", padx=8, pady=6,
                     anchor="center").grid(row=0, column=c, padx=1, pady=(0, 1), sticky="ew")

    def add_row(self, values, highlight=None):
        bg = self.ROW_BG[self._row_idx % 2]
        for c, (val, w) in enumerate(zip(values, self.col_widths)):
            fg = TEXT
            if highlight == "success": fg = SUCCESS
            elif highlight == "error": fg = ERROR
            elif highlight == "warn":  fg = WARNING
            tk.Label(self.inner, text=str(val), bg=bg, fg=fg,
                     font=("Segoe UI", 9), width=w // 9,
                     relief="flat", padx=8, pady=5,
                     anchor="center").grid(row=self._row_idx + 1, column=c,
                                           padx=1, pady=1, sticky="ew")
        self._row_idx += 1

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self._row_idx = 0
        self._draw_header()


# ─────────────────────────────────────────────────────────────────────────────
# Widget: tablero del puzzle
# ─────────────────────────────────────────────────────────────────────────────

class PuzzleBoard(tk.Canvas):
    """Dibuja un tablero n×n animado en un Canvas."""

    CELL = 72
    PAD  = 6

    def __init__(self, master, size=3, **kw):
        n = size
        total = n * self.CELL + (n + 1) * self.PAD
        super().__init__(master, width=total, height=total,
                         bg=BG, highlightthickness=0, **kw)
        self.size  = n
        self.board = None
        self.rects = {}
        self.texts = {}
        self._build_cells()

    def _build_cells(self):
        n, c, p = self.size, self.CELL, self.PAD
        for r in range(n):
            for col in range(n):
                x1 = col * (c + p) + p
                y1 = r   * (c + p) + p
                x2, y2 = x1 + c, y1 + c
                rid = self.create_rectangle(x1, y1, x2, y2,
                                            fill=TILE_CLR["empty"], outline="",
                                            width=0)
                tid = self.create_text((x1 + x2) // 2, (y1 + y2) // 2,
                                       text="", fill=TEXT,
                                       font=("Segoe UI", 18, "bold"))
                self.rects[(r, col)] = rid
                self.texts[(r, col)] = tid

    def set_board(self, board):
        """Actualiza visualmente el tablero (tuple lineal)."""
        self.board = board
        n = self.size
        for r in range(n):
            for col in range(n):
                val = board[r * n + col]
                rid = self.rects[(r, col)]
                tid = self.texts[(r, col)]
                if val == 0:
                    self.itemconfig(rid, fill=TILE_CLR["empty"])
                    self.itemconfig(tid, text="")
                else:
                    self.itemconfig(rid, fill=TILE_CLR["default"])
                    self.itemconfig(tid, text=str(val))


# ─────────────────────────────────────────────────────────────────────────────
# Widget: tablero de Sudoku
# ─────────────────────────────────────────────────────────────────────────────

class SudokuBoard(tk.Canvas):
    CELL = 52
    PAD  = 2

    def __init__(self, master, **kw):
        total = 9 * self.CELL + 10 * self.PAD + 6  # extra for 3×3 separators
        super().__init__(master, width=total, height=total,
                         bg=BG, highlightthickness=0, **kw)
        self.cells_bg  = {}
        self.cells_txt = {}
        self._build()

    def _build(self):
        c, p = self.CELL, self.PAD
        for r in range(9):
            for col in range(9):
                # extra offset for 3×3 box separators
                xoff = (col // 3) * 3
                yoff = (r   // 3) * 3
                x1 = col * (c + p) + p + xoff
                y1 = r   * (c + p) + p + yoff
                x2, y2 = x1 + c, y1 + c
                rid = self.create_rectangle(x1, y1, x2, y2,
                                            fill=TILE_CLR["empty"],
                                            outline=BORDER, width=1)
                tid = self.create_text((x1 + x2) // 2, (y1 + y2) // 2,
                                       text="", fill=TEXT,
                                       font=("Segoe UI", 14, "bold"))
                self.cells_bg[(r, col)]  = rid
                self.cells_txt[(r, col)] = tid
        # draw 3×3 box borders
        for i in range(4):
            xoff = i * (3 * (c + p) + 3)
            yoff = i * (3 * (c + p) + 3)
            self.create_line(xoff + p, p, xoff + p,
                             9*(c+p)+p+6, fill=ACCENT2, width=2)
            self.create_line(p, yoff + p, 9*(c+p)+p+6,
                             yoff + p, fill=ACCENT2, width=2)

    def set_board(self, board_2d, fixed_set=None):
        for r in range(9):
            for col in range(9):
                val = board_2d[r][col]
                rid = self.cells_bg[(r, col)]
                tid = self.cells_txt[(r, col)]
                if val == 0:
                    self.itemconfig(rid, fill=TILE_CLR["empty"])
                    self.itemconfig(tid, text="", fill=TEXT)
                else:
                    is_fixed = fixed_set and (r, col) in fixed_set
                    self.itemconfig(rid, fill=BG3 if is_fixed else TILE_CLR["default"])
                    self.itemconfig(tid, text=str(val),
                                    fill=ACCENT2 if is_fixed else SUCCESS)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Ejercicio 1 – Puzzle
# ─────────────────────────────────────────────────────────────────────────────

class PuzzleTab(tk.Frame):

    def __init__(self, master):
        super().__init__(master, bg=BG)
        self._build()

    def _build(self):
        # ── Left controls ────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG2, width=260)
        ctrl.pack(side="left", fill="y", padx=(16, 8), pady=16)
        ctrl.pack_propagate(False)

        _section(ctrl, "CONFIGURACIÓN")

        # puzzle size
        tk.Label(ctrl, text="Tamaño del puzzle", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(6, 2))
        self.size_var = tk.StringVar(value="8-puzzle")
        seg = _segmented(ctrl, ["8-puzzle", "15-puzzle"], self.size_var,
                         command=self._on_size_change)
        seg.pack(padx=12, fill="x")

        # difficulty
        tk.Label(ctrl, text="Dificultad", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(14, 2))
        self.diff_var = tk.StringVar(value="Fácil")
        seg2 = _segmented(ctrl, PUZZLE_LABELS, self.diff_var)
        seg2.pack(padx=12, fill="x")

        # heuristic
        tk.Label(ctrl, text="Heurística", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(14, 2))
        self.h_var = tk.StringVar(value=HEURISTICS[0][0])
        for name, _ in HEURISTICS:
            rb = tk.Radiobutton(ctrl, text=name, variable=self.h_var, value=name,
                                bg=BG2, fg=TEXT, selectcolor=BG3,
                                activebackground=BG2, activeforeground=ACCENT2,
                                font=("Segoe UI", 9), indicatoron=True)
            rb.pack(anchor="w", padx=16)

        # spacer
        tk.Frame(ctrl, bg=BG2, height=16).pack()

        # Run button
        self.run_btn = _btn(ctrl, "▶  Resolver", self._run, ACCENT)
        self.run_btn.pack(padx=12, fill="x", ipady=8)

        # Run all button
        tk.Frame(ctrl, bg=BG2, height=8).pack()
        self.all_btn = _btn(ctrl, "⚡ Comparar todo", self._run_all, BG3,
                            fg=ACCENT2)
        self.all_btn.pack(padx=12, fill="x", ipady=6)

        # Random button
        tk.Frame(ctrl, bg=BG2, height=8).pack()
        self.rand_btn = _btn(ctrl, "🎲 Aleatorio", self._randomize, BG3, fg=WARNING)
        self.rand_btn.pack(padx=12, fill="x", ipady=6)

        _section(ctrl, "RESULTADO")
        self.stat_frame = tk.Frame(ctrl, bg=BG2)
        self.stat_frame.pack(fill="x", padx=12)
        self._stats = {}
        for key, label in [("tiempo", "Tiempo"), ("memoria", "Memoria"),
                            ("nodos", "Nodos exp."), ("movimientos", "Movimientos")]:
            row = tk.Frame(self.stat_frame, bg=BG2)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=BG2, fg=TEXT_DIM,
                     font=("Segoe UI", 8), width=12, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text="–", bg=BG2, fg=TEXT,
                               font=("Segoe UI", 9, "bold"), anchor="w")
            val_lbl.pack(side="left")
            self._stats[key] = val_lbl

        self.solved_lbl = tk.Label(ctrl, text="", bg=BG2,
                                   font=("Segoe UI", 12, "bold"))
        self.solved_lbl.pack(pady=8)

        # ── Center: board ─────────────────────────────────────────────────────
        center = tk.Frame(self, bg=BG)
        center.pack(side="left", expand=True, fill="both", padx=8, pady=16)

        tk.Label(center, text="Tablero", bg=BG, fg=TEXT_DIM,
                 font=("Segoe UI", 10, "bold")).pack(pady=(0, 8))
        board_frame = tk.Frame(center, bg=BG)
        board_frame.pack()

        self.board_8  = PuzzleBoard(board_frame, size=3)
        self.board_15 = PuzzleBoard(board_frame, size=4)
        self.board_8.pack()
        self._active_board = self.board_8
        # Track current puzzle being displayed
        self._current_puzzle_8  = None
        self._current_puzzle_15 = None
        self._show_initial()

        # ── Right: comparison table ───────────────────────────────────────────
        right = tk.Frame(self, bg=BG2, width=610)
        right.pack(side="right", fill="y", padx=(8, 16), pady=16)
        right.pack_propagate(False)

        _section(right, "TABLA COMPARATIVA")
        self.table = StyledTable(right,
            headers=["Puzzle", "Dificultad", "Heurística", "Tiempo(s)",
                     "Mem(KB)", "Nodos", "Movs.", "✓"],
            col_widths=[75, 70, 165, 75, 65, 65, 55, 40])
        self.table.pack(fill="both", expand=True, padx=8, pady=8)

        self.progress = ttk.Progressbar(right, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=8, pady=(0, 8))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _on_size_change(self, val=None):
        size = self.size_var.get()
        if size == "8-puzzle":
            self.board_15.pack_forget()
            self.board_8.pack()
            self._active_board = self.board_8
        else:
            self.board_8.pack_forget()
            self.board_15.pack()
            self._active_board = self.board_15
        self._show_initial()

    def _show_initial(self):
        size = self.size_var.get()
        diff = PUZZLE_LABELS.index(self.diff_var.get())
        if size == "8-puzzle":
            p = PUZZLES_8[diff]
            self._current_puzzle_8 = p
            self._active_board.set_board(p)
        else:
            p = PUZZLES_15[diff]
            self._current_puzzle_15 = p
            self._active_board.set_board(p)

    def _get_heuristic_fn(self):
        name = self.h_var.get()
        return next(fn for n, fn in HEURISTICS if n == name)

    def _randomize(self):
        """Genera un puzzle aleatorio del tamaño actual y lo muestra."""
        import time as _time
        size = self.size_var.get()
        n = 3 if size == "8-puzzle" else 4
        # Use current milliseconds as seed for true randomness each click
        seed = int(_time.time() * 1000) % (2**31)
        puzzle = generate_random_puzzle(n, seed=seed)
        if size == "8-puzzle":
            self._current_puzzle_8 = puzzle
        else:
            self._current_puzzle_15 = puzzle
        self._active_board.set_board(puzzle)
        self.solved_lbl.config(text="", fg=TEXT)
        for k in self._stats:
            self._stats[k].config(text="–")

    def _run(self):
        self._set_running(True)
        size  = self.size_var.get()
        h_fn  = self._get_heuristic_fn()
        n     = 3 if size == "8-puzzle" else 4
        goal  = GOAL_8 if size == "8-puzzle" else GOAL_15

        # Use current displayed puzzle (may be random)
        if size == "8-puzzle":
            puzzle = self._current_puzzle_8 or PUZZLES_8[PUZZLE_LABELS.index(self.diff_var.get())]
        else:
            puzzle = self._current_puzzle_15 or PUZZLES_15[PUZZLE_LABELS.index(self.diff_var.get())]

        def task():
            result = a_star(puzzle, goal, n, h_fn)
            self.after(0, lambda: self._show_single_result(result, goal, n))

        threading.Thread(target=task, daemon=True).start()

    def _show_single_result(self, r, goal, n):
        self._active_board.set_board(goal)
        solved = r["solved"]
        to = r.get("timeout", False)
        self._stats["tiempo"].config(text=f"{r['time_s']:.4f} s")
        self._stats["memoria"].config(text=f"{r['mem_kb']:.1f} KB")
        self._stats["nodos"].config(text="TIMEOUT" if to else str(r["nodes"]))
        self._stats["movimientos"].config(text=str(r["moves"]) if solved else "–")
        if solved:
            self.solved_lbl.config(text="✓  Resuelto", fg=SUCCESS)
        elif to:
            self.solved_lbl.config(text="⏱  Tiempo agotado", fg=WARNING)
        else:
            self.solved_lbl.config(text="✗  Sin solución", fg=ERROR)
        self._set_running(False)

    def _run_all(self):
        self._set_running(True)
        self.table.clear()

        size  = self.size_var.get()
        n     = 3 if size == "8-puzzle" else 4
        goal  = GOAL_8 if size == "8-puzzle" else GOAL_15
        label = size
        diff  = self.diff_var.get()

        # Usar el tablero actualmente mostrado (puede ser aleatorio)
        if size == "8-puzzle":
            is_random = self._current_puzzle_8 is not None and \
                        self._current_puzzle_8 != PUZZLES_8[PUZZLE_LABELS.index(diff)]
            puzzle = self._current_puzzle_8 or PUZZLES_8[PUZZLE_LABELS.index(diff)]
        else:
            is_random = self._current_puzzle_15 is not None and \
                        self._current_puzzle_15 != PUZZLES_15[PUZZLE_LABELS.index(diff)]
            puzzle = self._current_puzzle_15 or PUZZLES_15[PUZZLE_LABELS.index(diff)]

        diff_label = "Aleatorio" if is_random else diff

        def task():
            for h_name, h_fn in HEURISTICS:
                r = a_star(puzzle, goal, n, h_fn)
                to = r.get("timeout", False)
                row = [
                    label,
                    diff_label,          # ← columna Dificultad (faltaba)
                    h_name[:24],
                    f"{r['time_s']:.6f}",
                    f"{r['mem_kb']:.1f}",
                    "TIMEOUT" if to else r["nodes"],
                    r["moves"] if r["solved"] else "–",
                    "TIMEOUT" if to else ("✓" if r["solved"] else "✗"),
                ]
                hl = "success" if r["solved"] else ("warn" if to else "error")
                self.after(0, lambda rx=row, h=hl: self.table.add_row(rx, h))
            self.after(0, lambda: self._set_running(False))

        threading.Thread(target=task, daemon=True).start()

    def _set_running(self, running: bool):
        state = "disabled" if running else "normal"
        self.run_btn.config(state=state)
        self.all_btn.config(state=state)
        self.rand_btn.config(state=state)
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Ejercicio 2 – Sudoku
# ─────────────────────────────────────────────────────────────────────────────

class SudokuTab(tk.Frame):

    def __init__(self, master):
        super().__init__(master, bg=BG)
        self._build()

    def _build(self):
        # ── Left controls ─────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG2, width=260)
        ctrl.pack(side="left", fill="y", padx=(16, 8), pady=16)
        ctrl.pack_propagate(False)

        _section(ctrl, "CONFIGURACIÓN")

        tk.Label(ctrl, text="Nivel de dificultad", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(6, 4))
        self.level_var = tk.IntVar(value=0)
        for i, lbl in enumerate(SUDOKU_LABELS):
            rb = tk.Radiobutton(ctrl, text=lbl, variable=self.level_var, value=i,
                                command=self._show_initial,
                                bg=BG2, fg=TEXT, selectcolor=BG3,
                                activebackground=BG2, activeforeground=ACCENT2,
                                font=("Segoe UI", 9))
            rb.pack(anchor="w", padx=16)

        tk.Label(ctrl, text="Algoritmo", bg=BG2, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(14, 4))
        self.algo_var = tk.StringVar(value="A*")
        for alg in ["A*", "Recocido Simulado", "Ambos"]:
            rb = tk.Radiobutton(ctrl, text=alg, variable=self.algo_var, value=alg,
                                bg=BG2, fg=TEXT, selectcolor=BG3,
                                activebackground=BG2, activeforeground=ACCENT2,
                                font=("Segoe UI", 9))
            rb.pack(anchor="w", padx=16)

        tk.Frame(ctrl, bg=BG2, height=16).pack()
        self.run_btn = _btn(ctrl, "▶  Resolver", self._run, ACCENT)
        self.run_btn.pack(padx=12, fill="x", ipady=8)
        tk.Frame(ctrl, bg=BG2, height=8).pack()
        self.all_btn = _btn(ctrl, "⚡ Comparar todo", self._run_all, BG3, fg=ACCENT2)
        self.all_btn.pack(padx=12, fill="x", ipady=6)
        tk.Frame(ctrl, bg=BG2, height=8).pack()
        self.rand_btn = _btn(ctrl, "🎲 Aleatorio", self._randomize_sudoku, BG3, fg=WARNING)
        self.rand_btn.pack(padx=12, fill="x", ipady=6)

        _section(ctrl, "RESULTADO")
        self.stat_frame = tk.Frame(ctrl, bg=BG2)
        self.stat_frame.pack(fill="x", padx=12)
        self._stats = {}
        for key, label in [("tiempo", "Tiempo"), ("memoria", "Memoria"),
                            ("iters", "Nodos/Iters")]:
            row = tk.Frame(self.stat_frame, bg=BG2)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=BG2, fg=TEXT_DIM,
                     font=("Segoe UI", 8), width=12, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text="–", bg=BG2, fg=TEXT,
                               font=("Segoe UI", 9, "bold"), anchor="w")
            val_lbl.pack(side="left")
            self._stats[key] = val_lbl

        self.solved_lbl = tk.Label(ctrl, text="", bg=BG2,
                                   font=("Segoe UI", 12, "bold"))
        self.solved_lbl.pack(pady=8)

        # ── Center: sudoku board ───────────────────────────────────────────────
        center = tk.Frame(self, bg=BG)
        center.pack(side="left", expand=True, fill="both", padx=8, pady=16)

        self.board_label = tk.Label(center, text="Tablero Inicial", bg=BG,
                                    fg=TEXT_DIM, font=("Segoe UI", 10, "bold"))
        self.board_label.pack(pady=(0, 8))
        self.board = SudokuBoard(center)
        self.board.pack()
        self._fixed = set()
        self._current_sudoku = None  # tracks randomly generated board
        self._show_initial()

        # ── Right: table ───────────────────────────────────────────────────────
        right = tk.Frame(self, bg=BG2, width=560)
        right.pack(side="right", fill="y", padx=(8, 16), pady=16)
        right.pack_propagate(False)

        _section(right, "TABLA COMPARATIVA")
        self.table = StyledTable(right,
            headers=["Nivel", "Algoritmo", "Tiempo(s)", "Mem(KB)",
                     "Nodos/Iters", "Resuelto"],
            col_widths=[155, 130, 75, 70, 90, 65])
        self.table.pack(fill="both", expand=True, padx=8, pady=8)

        self.progress = ttk.Progressbar(right, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=8, pady=(0, 8))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _show_initial(self):
        puzzle = SUDOKU_PUZZLES[self.level_var.get()]
        self._current_sudoku = puzzle
        self._fixed = {(r, c) for r in range(9) for c in range(9)
                       if puzzle[r][c] != 0}
        self.board.set_board(puzzle, self._fixed)
        self.board_label.config(text="Tablero Inicial")

    def _randomize_sudoku(self):
        """Genera un sudoku aleatorio con las celdas vacías del nivel actual."""
        import time as _time
        empty_counts = [20, 35, 45]
        n_empty = empty_counts[self.level_var.get()]
        seed = int(_time.time() * 1000) % (2**31)
        puzzle = generate_random_sudoku(n_empty, seed=seed)
        self._current_sudoku = puzzle
        self._fixed = {(r, c) for r in range(9) for c in range(9)
                       if puzzle[r][c] != 0}
        self.board.set_board(puzzle, self._fixed)
        self.board_label.config(text="Tablero Aleatorio")
        self.solved_lbl.config(text="", fg=TEXT)
        for k in self._stats:
            self._stats[k].config(text="–")

    def _run(self):
        self._set_running(True)
        # Use current displayed puzzle (may be random)
        puzzle = self._current_sudoku if self._current_sudoku is not None \
                 else SUDOKU_PUZZLES[self.level_var.get()]
        algo   = self.algo_var.get()

        def task():
            if algo in ("A*", "Ambos"):
                r = a_star_sudoku(puzzle)
                self.after(0, lambda: self._show_result(r, puzzle, "A*"))
            if algo in ("Recocido Simulado", "Ambos"):
                r2 = simulated_annealing(puzzle)
                self.after(0, lambda: self._show_result(r2, puzzle, "RS"))

        threading.Thread(target=task, daemon=True).start()

    def _show_result(self, r, puzzle, algo_tag):
        self._stats["tiempo"].config(text=f"{r['time_s']:.4f} s")
        self._stats["memoria"].config(text=f"{r['mem_kb']:.1f} KB")
        iters = r.get("nodes") or r.get("iterations") or "–"
        self._stats["iters"].config(text=str(iters))
        if r["solved"]:
            self.solved_lbl.config(
                text=f"✓  {algo_tag} – Resuelto", fg=SUCCESS)
            # show completed board visually (approximation: show goal)
            self.board_label.config(text="Tablero Resuelto")
            from sudoku import _SOLUTION
            self.board.set_board(_SOLUTION, self._fixed)
        else:
            self.solved_lbl.config(text="✗  No resuelto", fg=ERROR)
        self._set_running(False)

    def _run_all(self):
        self._set_running(True)
        self.table.clear()

        # Usar el sudoku actualmente mostrado (puede ser aleatorio)
        puzzle = self._current_sudoku if self._current_sudoku is not None \
                 else SUDOKU_PUZZLES[self.level_var.get()]
        level_labels = ["Fácil (20)", "Intermedio (35)", "Difícil (45)"]
        level_name = level_labels[self.level_var.get()]

        def task():
            r_astar = a_star_sudoku(puzzle)
            r_sa    = simulated_annealing(puzzle)
            for r, alg in [(r_astar, "A*"), (r_sa, "Recocido Simulado")]:
                iters = r.get("nodes") or r.get("iterations") or "–"
                row = [level_name, alg,
                       f"{r['time_s']:.6f}",
                       f"{r['mem_kb']:.1f}",
                       iters, "✓" if r["solved"] else "✗"]
                hl = "success" if r["solved"] else "error"
                self.after(0, lambda rx=row, h=hl: self.table.add_row(rx, h))
            self.after(0, lambda: self._set_running(False))

        threading.Thread(target=task, daemon=True).start()

    def _set_running(self, running: bool):
        state = "disabled" if running else "normal"
        self.run_btn.config(state=state)
        self.all_btn.config(state=state)
        self.rand_btn.config(state=state)
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Información / Ayuda
# ─────────────────────────────────────────────────────────────────────────────

class InfoTab(tk.Frame):

    CONTENT = """
  ALGORITMO A*
  ─────────────────────────────────────────────────────────────
  A* es un algoritmo de búsqueda informada que combina el
  coste real g(n) con una heurística h(n) para guiar la
  búsqueda. La función de evaluación es:

        f(n) = g(n) + h(n)

  Si h(n) es admisible (nunca sobreestima el coste real),
  A* garantiza encontrar la solución óptima.

  ─────────────────────────────────────────────────────────────
  HEURÍSTICAS – EJERCICIO 1
  ─────────────────────────────────────────────────────────────

  1. Fichas fuera de lugar
     Cuenta cuántas fichas no están en su posición meta.
     Admisible, pero poco informada → más nodos expandidos.

  2. Distancia Manhattan
     Suma la distancia horizontal + vertical de cada ficha
     a su posición meta. Más informada, menos nodos.

  3. Heurística personalizada
     Manhattan + 2 × conflictos lineales.
     Dos fichas tienen conflicto lineal si están en la misma
     fila/columna meta pero en orden inverso.
     Domina a Manhattan → aún menos nodos. Son admisibles.

  ─────────────────────────────────────────────────────────────
  HEURÍSTICAS – EJERCICIO 2 (Sudoku)
  ─────────────────────────────────────────────────────────────

  A* con MRV (Minimum Remaining Values):
    g = celdas rellenadas; h = mínimo de candidatos válidos
    entre las celdas vacías restantes. Siempre elige la celda
    con menos candidatos (exploración más eficiente).

  ─────────────────────────────────────────────────────────────
  RECOCIDO SIMULADO
  ─────────────────────────────────────────────────────────────
  Metaheurística inspirada en el enfriamiento de metales.
  Comienza con una solución aleatoria y acepta soluciones
  peores con probabilidad exp(-ΔE/T), decreciente con T.

  • Temperatura inicial T₀ = 2.0
  • Factor de enfriamiento α = 0.9999
  • Temperatura mínima T_min = 0.001
  • Máximo de iteraciones: 500 000

  La energía es el número de conflictos en filas y columnas.
"""

    def __init__(self, master):
        super().__init__(master, bg=BG)
        frame = tk.Frame(self, bg=BG2, padx=32, pady=24)
        frame.pack(expand=True, fill="both", padx=32, pady=32)
        tk.Label(frame, text="📖  Información Teórica", bg=BG2, fg=ACCENT2,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 12))
        txt = tk.Text(frame, bg=BG2, fg=TEXT, font=("Courier New", 10),
                      relief="flat", wrap="word", state="normal",
                      padx=8, pady=8, spacing1=2)
        txt.insert("1.0", self.CONTENT)
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ─────────────────────────────────────────────────────────────────────────────

def _section(parent, title):
    tk.Label(parent, text=title, bg=BG2, fg=ACCENT2,
             font=("Segoe UI", 8, "bold"), anchor="w").pack(
        anchor="w", padx=12, pady=(16, 4))
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 6))


def _btn(parent, text, command, bg, fg=TEXT, **kw):
    btn = tk.Button(parent, text=text, command=command,
                    bg=bg, fg=fg, relief="flat", cursor="hand2",
                    activebackground=ACCENT2, activeforeground=BG,
                    font=("Segoe UI", 10, "bold"), **kw)

    def _on_enter(e):
        btn.config(bg=ACCENT2, fg=BG)

    def _on_leave(e):
        btn.config(bg=bg, fg=fg)

    btn.bind("<Enter>", _on_enter)
    btn.bind("<Leave>", _on_leave)
    return btn


class _SegBtn(tk.Button):
    pass


def _segmented(parent, options, var, command=None):
    """Grupo de botones tipo segmented control."""
    frame = tk.Frame(parent, bg=BG3, padx=2, pady=2)

    def _mk(opt):
        def cb():
            var.set(opt)
            _refresh()
            if command:
                command(opt)

        btn = tk.Button(frame, text=opt, command=cb,
                        relief="flat", cursor="hand2",
                        font=("Segoe UI", 8, "bold"),
                        padx=6, pady=4)
        btn.pack(side="left", fill="x", expand=True)
        return btn

    btns = [_mk(opt) for opt in options]

    def _refresh():
        for btn, opt in zip(btns, options):
            if var.get() == opt:
                btn.config(bg=ACCENT, fg=TEXT)
            else:
                btn.config(bg=BG3, fg=TEXT_DIM,
                           activebackground=BG3, activeforeground=TEXT)

    _refresh()
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Header bar (logo + title)
# ─────────────────────────────────────────────────────────────────────────────

class HeaderBar(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG2, height=64)
        self.pack_propagate(False)

        left = tk.Frame(self, bg=BG2)
        left.pack(side="left", padx=20)

        tk.Label(left, text="🧩", bg=BG2, font=("Segoe UI", 22)).pack(
            side="left", padx=(0, 10))
        tk.Label(left, text="Práctica 2 – Búsqueda Heurística",
                 bg=BG2, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(left, text="  A*  ·  Recocido Simulado",
                 bg=BG2, fg=TEXT_DIM, font=("Segoe UI", 10)).pack(side="left")

        right = tk.Frame(self, bg=BG2)
        right.pack(side="right", padx=20)
        tk.Label(right, text="Python 3  ·  tkinter", bg=BG2,
                 fg=TEXT_DIM, font=("Segoe UI", 8)).pack()


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Práctica 2 – Búsqueda Heurística")
        self.geometry("1280x780")
        self.minsize(1100, 680)
        self.configure(bg=BG)
        self._setup_styles()
        self._build()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",
                        background=BG, borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab",
                        background=BG2, foreground=TEXT_DIM,
                        padding=[20, 10], font=("Segoe UI", 10, "bold"),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG3), ("active", BG3)],
                  foreground=[("selected", ACCENT2), ("active", TEXT)])
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=BG3, bordercolor=BG3,
                        background=ACCENT, lightcolor=ACCENT2,
                        darkcolor=ACCENT)

    def _build(self):
        HeaderBar(self).pack(fill="x")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        nb = ttk.Notebook(self, style="TNotebook")
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.puzzle_tab = PuzzleTab(nb)
        self.sudoku_tab = SudokuTab(nb)
        self.info_tab   = InfoTab(nb)

        nb.add(self.puzzle_tab, text="  🧩  Ej 1 – Puzzle  ")
        nb.add(self.sudoku_tab, text="  🔢  Ej 2 – Sudoku  ")
        nb.add(self.info_tab,   text="  📖  Información    ")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
