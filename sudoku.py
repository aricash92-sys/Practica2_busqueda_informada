"""
Ejercicio 2: Sudoku resuelto con A* y Recocido Simulado.
Niveles: Fácil (20 celdas vacías), Intermedio (35), Difícil (45).
"""

import heapq
import math
import random
import time
import tracemalloc
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
import copy

# ---------------------------------------------------------------------------
# Puzzles de Sudoku predefinidos
# ---------------------------------------------------------------------------
# 0 = celda vacía. Los puzzles son válidos y tienen solución única.

_BASE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# Solución del _BASE
_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def _count_empty(board):
    return sum(row.count(0) for row in board)


def _remove_cells(solution, n_remove, seed=42):
    """Elimina n_remove celdas de la solución de forma determinista."""
    rng = random.Random(seed)
    board = [row[:] for row in solution]
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    removed = 0
    for r, c in cells:
        if removed == n_remove:
            break
        board[r][c] = 0
        removed += 1
    return board


# Generar los tres niveles a partir de la solución conocida
EASY   = _remove_cells(_SOLUTION, 20, seed=1)   # 20 celdas vacías
MEDIUM = _remove_cells(_SOLUTION, 35, seed=2)   # 35 celdas vacías
HARD   = _remove_cells(_SOLUTION, 45, seed=3)   # 45 celdas vacías

LEVELS = [("Fácil (20 vacías)", EASY), ("Intermedio (35 vacías)", MEDIUM), ("Difícil (45 vacías)", HARD)]

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def board_to_tuple(board):
    return tuple(tuple(row) for row in board)


def tuple_to_board(t):
    return [list(row) for row in t]


def get_empty_cells(board):
    return [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]


def is_valid(board, row, col, num):
    """Devuelve True si num puede colocarse en (row, col) según las reglas del Sudoku."""
    # Fila
    if num in board[row]:
        return False
    # Columna
    if num in [board[r][col] for r in range(9)]:
        return False
    # Caja 3×3
    br, bc = (row // 3) * 3, (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == num:
                return False
    return True


def get_candidates(board, row, col):
    """Retorna el conjunto de valores válidos para la celda (row, col)."""
    if board[row][col] != 0:
        return set()
    used = set(board[row])                                    # fila
    used |= {board[r][col] for r in range(9)}                 # columna
    br, bc = (row // 3) * 3, (col // 3) * 3
    used |= {board[r][c] for r in range(br, br+3) for c in range(bc, bc+3)}
    return set(range(1, 10)) - used


# ---------------------------------------------------------------------------
# A* para Sudoku
# ---------------------------------------------------------------------------
#
# Representación: tablero como tuple-of-tuples (inmutable, hashable).
# g = número de celdas rellenadas desde el inicio.
# h = MRV: mín. de candidatos válidos entre las celdas vacías restantes.
#     Si alguna celda queda sin candidatos → estado inválido, h = ∞.
# Se elige siempre la celda vacía con MENOS candidatos (MRV branching).

@dataclass(order=True)
class SudokuNode:
    f: int
    g: int = field(compare=False)
    board: tuple = field(compare=False)   # tuple-of-tuples

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return self.board == other.board


def _mrv_heuristic(board_list):
    """MRV: suma del mínimo número de candidatos de cada celda vacía.
    Si alguna celda no tiene candidatos, devuelve float('inf')."""
    empty = get_empty_cells(board_list)
    if not empty:
        return 0
    min_cands = float("inf")
    for r, c in empty:
        cands = get_candidates(board_list, r, c)
        if not cands:
            return float("inf")
        if len(cands) < min_cands:
            min_cands = len(cands)
    return min_cands


def _pick_mrv_cell(board_list):
    """Selecciona la celda vacía con menos candidatos (MRV)."""
    empty = get_empty_cells(board_list)
    best_cell = None
    best_count = float("inf")
    best_cands = set()
    for r, c in empty:
        cands = get_candidates(board_list, r, c)
        if len(cands) < best_count:
            best_count = len(cands)
            best_cell = (r, c)
            best_cands = cands
    return best_cell, best_cands


def a_star_sudoku(puzzle):
    """
    Resuelve el sudoku con A* (MRV).
    Devuelve dict con solved, time_s, mem_kb, nodes.
    """
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    t0 = time.perf_counter()

    board_list = [row[:] for row in puzzle]
    start_board = board_to_tuple(board_list)
    fixed_count = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)

    h0 = _mrv_heuristic(board_list)
    if h0 == float("inf"):
        if not already_tracing:
            tracemalloc.stop()
        return {"solved": False, "time_s": 0, "mem_kb": 0, "nodes": 0}

    start_node = SudokuNode(f=0 + h0, g=0, board=start_board)
    open_heap = [start_node]
    visited = set()
    nodes_expanded = 0

    while open_heap:
        node = heapq.heappop(open_heap)

        if node.board in visited:
            continue
        visited.add(node.board)
        nodes_expanded += 1

        current_list = tuple_to_board(node.board)
        empty = get_empty_cells(current_list)

        if not empty:
            # ¡Resuelto!
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            if not already_tracing:
                tracemalloc.stop()
            return {"solved": True, "time_s": elapsed,
                    "mem_kb": peak / 1024, "nodes": nodes_expanded}

        cell, cands = _pick_mrv_cell(current_list)
        if cell is None or not cands:
            continue

        r, c = cell
        for num in sorted(cands):
            new_list = [row[:] for row in current_list]
            new_list[r][c] = num
            new_board = board_to_tuple(new_list)
            if new_board in visited:
                continue
            h_val = _mrv_heuristic(new_list)
            if h_val == float("inf"):
                continue
            new_g = node.g + 1
            new_node = SudokuNode(f=new_g + h_val, g=new_g, board=new_board)
            heapq.heappush(open_heap, new_node)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    if not already_tracing:
        tracemalloc.stop()
    return {"solved": False, "time_s": elapsed, "mem_kb": peak / 1024, "nodes": nodes_expanded}


# ---------------------------------------------------------------------------
# Recocido Simulado para Sudoku
# ---------------------------------------------------------------------------
#
# Inicialización: para cada caja 3×3, rellenar las celdas vacías con los
# números faltantes en esa caja (sin repetición dentro de la caja).
# Energía: número total de conflictos en filas + columnas.
# Movimiento: intercambiar dos celdas NO fijas dentro de la misma caja.

def _init_sa(puzzle):
    """Rellena las cajas con los números faltantes al azar."""
    rng = random.Random(7)
    board = [row[:] for row in puzzle]
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            present = {board[r][c]
                       for r in range(br, br+3)
                       for c in range(bc, bc+3)
                       if board[r][c] != 0}
            missing = list(set(range(1, 10)) - present)
            rng.shuffle(missing)
            idx = 0
            for r in range(br, br+3):
                for c in range(bc, bc+3):
                    if board[r][c] == 0:
                        board[r][c] = missing[idx]
                        idx += 1
    return board


def _energy(board):
    """Número de conflictos en filas + columnas (duplicados)."""
    conflicts = 0
    for i in range(9):
        row_vals = [board[i][c] for c in range(9)]
        conflicts += 9 - len(set(row_vals))
        col_vals = [board[r][i] for r in range(9)]
        conflicts += 9 - len(set(col_vals))
    return conflicts


def simulated_annealing(puzzle, max_iter=500_000, T_init=2.0,
                        alpha=0.9999, T_min=0.001, seed=42):
    """
    Resuelve el Sudoku con Recocido Simulado.
    Devuelve dict con solved, time_s, mem_kb, iterations.
    """
    rng = random.Random(seed)
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    t0 = time.perf_counter()

    # Celdas fijas (no se pueden mover)
    fixed = {(r, c) for r in range(9) for c in range(9) if puzzle[r][c] != 0}

    board = _init_sa(puzzle)
    E = _energy(board)

    T = T_init
    iterations = 0

    for iterations in range(1, max_iter + 1):
        if E == 0:
            break

        # Elegir una caja al azar con al menos 2 celdas no fijas
        box_candidates = []
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                non_fixed = [(r, c) for r in range(br, br+3)
                             for c in range(bc, bc+3) if (r, c) not in fixed]
                if len(non_fixed) >= 2:
                    box_candidates.append(non_fixed)

        if not box_candidates:
            break

        non_fixed = rng.choice(box_candidates)
        r1, c1 = rng.choice(non_fixed)
        r2, c2 = rng.choice(non_fixed)
        while (r1, c1) == (r2, c2):
            r2, c2 = rng.choice(non_fixed)

        # Intercambio
        board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
        new_E = _energy(board)
        delta = new_E - E

        if delta < 0 or rng.random() < math.exp(-delta / T):
            E = new_E
        else:
            # Revertir
            board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]

        T = max(T * alpha, T_min)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    if not already_tracing:
        tracemalloc.stop()

    return {
        "solved": E == 0,
        "time_s": elapsed,
        "mem_kb": peak / 1024,
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Generador de Sudokus aleatorios
# ---------------------------------------------------------------------------

def generate_random_sudoku(n_empty: int, seed: int = None) -> list:
    """
    Genera un sudoku aleatorio con n_empty celdas vacías.
    Toma _SOLUTION como base, aplica permutaciones válidas (intercambio de
    filas/columnas dentro del mismo bloque de 3) para obtener un tablero
    diferente cada vez, y luego elimina n_empty celdas.
    """
    rng = random.Random(seed)
    board = copy.deepcopy(_SOLUTION)

    # Permutación de filas dentro de cada banda horizontal (3 bandas de 3 filas)
    for band in range(3):
        rows = list(range(band * 3, band * 3 + 3))
        rng.shuffle(rows)
        new_rows = [board[r][:] for r in rows]
        for i, r in enumerate(range(band * 3, band * 3 + 3)):
            board[r] = new_rows[i]

    # Permutación de columnas dentro de cada banda vertical (3 bandas de 3 cols)
    for band in range(3):
        cols = list(range(band * 3, band * 3 + 3))
        rng.shuffle(cols)
        new_board = copy.deepcopy(board)
        for i, c_dst in enumerate(range(band * 3, band * 3 + 3)):
            c_src = cols[i]
            for r in range(9):
                new_board[r][c_dst] = board[r][c_src]
        board = new_board

    # Intercambio de bandas horizontales completas
    band_order = [0, 1, 2]
    rng.shuffle(band_order)
    reordered = []
    for b in band_order:
        for r in range(b * 3, b * 3 + 3):
            reordered.append(board[r][:])
    board = reordered

    # Eliminar n_empty celdas
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    for r, c in cells[:n_empty]:
        board[r][c] = 0

    return board


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_sudoku_comparison():
    """Ejecuta A* y SA sobre los tres niveles y devuelve lista de dicts."""
    results = []
    for level_name, puzzle in LEVELS:
        # A*
        r = a_star_sudoku(puzzle)
        r["algorithm"] = "A*"
        r["level"] = level_name
        r.setdefault("iterations", r.get("nodes", "–"))
        results.append(r)

        # Recocido Simulado
        r2 = simulated_annealing(puzzle)
        r2["algorithm"] = "Recocido Simulado"
        r2["level"] = level_name
        r2.setdefault("nodes", r2.get("iterations", "–"))
        results.append(r2)

    return results


if __name__ == "__main__":
    from main import print_sudoku_table
    print_sudoku_table(run_sudoku_comparison())
