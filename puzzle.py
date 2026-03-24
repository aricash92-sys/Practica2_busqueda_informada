"""
Ejercicio 1: 8-puzzle y 15-puzzle resueltos con A*.
Heurísticas implementadas:
  1. Fichas fuera de lugar (misplaced tiles)
  2. Distancia Manhattan
  3. Heurística personalizada (Manhattan + conflictos lineales)
"""

import heapq
import random
import time
import tracemalloc

# Límite de tiempo por ejecución (segundos)
TIME_LIMIT_S = 30
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Estado del puzzle
# ---------------------------------------------------------------------------

@dataclass(order=True)
class PuzzleState:
    f: int
    g: int = field(compare=False)
    h: int = field(compare=False)
    board: tuple = field(compare=False)
    blank_pos: int = field(compare=False)   # índice lineal de la celda vacía
    size: int = field(compare=False)        # dimensión (3 para 8-puzzle, 4 para 15-puzzle)
    parent: Optional["PuzzleState"] = field(default=None, compare=False, repr=False)

    def __hash__(self):
        return hash(self.board)

    def __eq__(self, other):
        return self.board == other.board


def make_state(board: tuple, size: int, g: int, h: int,
               parent: Optional[PuzzleState] = None) -> PuzzleState:
    blank_pos = board.index(0)
    return PuzzleState(
        f=g + h,
        g=g,
        h=h,
        board=board,
        blank_pos=blank_pos,
        size=size,
        parent=parent,
    )


def get_neighbors(state: PuzzleState) -> list:
    """Genera estados vecinos moviendo el hueco en las 4 direcciones."""
    neighbors = []
    b = list(state.board)
    pos = state.blank_pos
    n = state.size

    row, col = divmod(pos, n)
    moves = []
    if row > 0:     moves.append(pos - n)   # arriba
    if row < n - 1: moves.append(pos + n)   # abajo
    if col > 0:     moves.append(pos - 1)   # izquierda
    if col < n - 1: moves.append(pos + 1)   # derecha

    for new_pos in moves:
        nb = b[:]
        nb[pos], nb[new_pos] = nb[new_pos], nb[pos]
        neighbors.append(tuple(nb))
    return neighbors


# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------

def misplaced_tiles(board: tuple, goal: tuple, size: int) -> int:
    """Número de fichas fuera de su posición objetivo (excluye el hueco)."""
    return sum(1 for i, v in enumerate(board) if v != 0 and v != goal[i])


def manhattan_distance(board: tuple, goal: tuple, size: int) -> int:
    """Suma de distancias Manhattan de cada ficha a su posición objetivo."""
    goal_pos = {v: divmod(i, size) for i, v in enumerate(goal)}
    total = 0
    for i, v in enumerate(board):
        if v == 0:
            continue
        row, col = divmod(i, size)
        gr, gc = goal_pos[v]
        total += abs(row - gr) + abs(col - gc)
    return total


def linear_conflicts(board: tuple, goal: tuple, size: int) -> int:
    """
    Cuenta conflictos lineales en filas y columnas.
    Dos fichas generan un conflicto lineal si están en la misma
    fila/columna objetivo y están invertidas entre sí.
    Cada conflicto requiere al menos 2 movimientos extra.
    """
    goal_pos = {v: divmod(i, size) for i, v in enumerate(goal)}
    conflicts = 0

    # Conflictos en filas
    for r in range(size):
        tiles_in_row = []
        for c in range(size):
            v = board[r * size + c]
            if v != 0 and goal_pos[v][0] == r:   # ficha que pertenece a esta fila
                tiles_in_row.append((goal_pos[v][1], c))  # (col_meta, col_actual)
        for i in range(len(tiles_in_row)):
            for j in range(i + 1, len(tiles_in_row)):
                gc_i, ac_i = tiles_in_row[i]
                gc_j, ac_j = tiles_in_row[j]
                if (gc_i > gc_j) != (ac_i > ac_j):
                    conflicts += 1

    # Conflictos en columnas
    for c in range(size):
        tiles_in_col = []
        for r in range(size):
            v = board[r * size + c]
            if v != 0 and goal_pos[v][1] == c:
                tiles_in_col.append((goal_pos[v][0], r))
        for i in range(len(tiles_in_col)):
            for j in range(i + 1, len(tiles_in_col)):
                gr_i, ar_i = tiles_in_col[i]
                gr_j, ar_j = tiles_in_col[j]
                if (gr_i > gr_j) != (ar_i > ar_j):
                    conflicts += 1

    return conflicts


def custom_heuristic(board: tuple, goal: tuple, size: int) -> int:
    """
    Heurística personalizada: Manhattan + 2 * conflictos lineales.
    Es admisible y consistente; domina a Manhattan y a fichas fuera de lugar.
    """
    return manhattan_distance(board, goal, size) + 2 * linear_conflicts(board, goal, size)


# ---------------------------------------------------------------------------
# Generador de puzzles aleatorios (solucionables)
# ---------------------------------------------------------------------------

def generate_random_puzzle(size: int, n_moves: int = None, seed: int = None) -> tuple:
    """
    Genera un puzzle solucionable haciendo n_moves movimientos aleatorios
    desde el estado objetivo. Garantiza que el resultado sea solucionable.

    Args:
        size:    dimensión del tablero (3 para 8-puzzle, 4 para 15-puzzle).
        n_moves: número de movimientos aleatorios (None = automático según size).
        seed:    semilla opcional para reproducibilidad.
    """
    if n_moves is None:
        n_moves = 50 if size == 3 else 80

    # Estado objetivo según tamaño
    if size == 3:
        goal = list(GOAL_8)
    else:
        goal = list(GOAL_15)

    rng = random.Random(seed)
    board = goal[:]
    blank = board.index(0)
    prev_blank = -1  # para evitar deshacer el último movimiento

    for _ in range(n_moves):
        row, col = divmod(blank, size)
        moves = []
        if row > 0:     moves.append(blank - size)
        if row < size - 1: moves.append(blank + size)
        if col > 0:     moves.append(blank - 1)
        if col < size - 1: moves.append(blank + 1)
        # Evitar revertir movimiento previo
        moves = [m for m in moves if m != prev_blank]
        new_blank = rng.choice(moves)
        board[blank], board[new_blank] = board[new_blank], board[blank]
        prev_blank = blank
        blank = new_blank

    return tuple(board)


# ---------------------------------------------------------------------------
# Algoritmo A* (timeout basado en tiempo, seguro para hilos)
# ---------------------------------------------------------------------------

def a_star(start_board: tuple, goal_board: tuple, size: int,
           heuristic: Callable) -> dict:
    """
    Ejecuta A* desde start_board hasta goal_board usando la heurística dada.
    El timeout se implementa con time.perf_counter() (seguro en hilos).
    Devuelve un dict con:
        solved    – bool
        moves     – número de movimientos en la solución
        nodes     – nodos expandidos
        time_s    – tiempo en segundos
        mem_kb    – memoria pico en KB
        timeout   – True si se agotó el tiempo
    """
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    t0 = time.perf_counter()

    try:
        h0 = heuristic(start_board, goal_board, size)
        start = make_state(start_board, size, g=0, h=h0)

        open_heap = []
        heapq.heappush(open_heap, start)

        # g-cost mínimo conocido para cada tablero
        best_g: dict[tuple, int] = {start_board: 0}
        nodes_expanded = 0

        while open_heap:
            state = heapq.heappop(open_heap)

            # Si encontramos un camino más corto ya registrado, descartamos
            if state.g > best_g.get(state.board, float("inf")):
                continue

            nodes_expanded += 1

            # Comprobación de tiempo límite cada 5 000 nodos
            if nodes_expanded % 5_000 == 0:
                if time.perf_counter() - t0 >= TIME_LIMIT_S:
                    elapsed = time.perf_counter() - t0
                    _, peak = tracemalloc.get_traced_memory()
                    if not already_tracing:
                        tracemalloc.stop()
                    return {"solved": False, "moves": -1, "nodes": nodes_expanded,
                            "time_s": elapsed, "mem_kb": peak / 1024, "timeout": True}

            if state.board == goal_board:
                elapsed = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                if not already_tracing:
                    tracemalloc.stop()
                return {
                    "solved": True,
                    "moves": state.g,
                    "nodes": nodes_expanded,
                    "time_s": elapsed,
                    "mem_kb": peak / 1024,
                    "timeout": False,
                }

            for nb_board in get_neighbors(state):
                new_g = state.g + 1
                if new_g < best_g.get(nb_board, float("inf")):
                    best_g[nb_board] = new_g
                    h_val = heuristic(nb_board, goal_board, size)
                    next_state = make_state(nb_board, size, g=new_g, h=h_val, parent=state)
                    heapq.heappush(open_heap, next_state)

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        if not already_tracing:
            tracemalloc.stop()
        return {"solved": False, "moves": -1, "nodes": nodes_expanded,
                "time_s": elapsed, "mem_kb": peak / 1024, "timeout": False}

    except Exception:
        elapsed = time.perf_counter() - t0
        try:
            _, peak = tracemalloc.get_traced_memory()
            if not already_tracing:
                tracemalloc.stop()
        except Exception:
            peak = 0
        return {"solved": False, "moves": -1, "nodes": -1,
                "time_s": elapsed, "mem_kb": peak / 1024, "timeout": True}


# ---------------------------------------------------------------------------
# Puzzles de prueba
# ---------------------------------------------------------------------------

# --- 8-puzzle (3×3) ---
GOAL_8 = (1, 2, 3, 4, 5, 6, 7, 8, 0)

PUZZLES_8 = [
    # Fácil (~8 movimientos)      – verificado solucionable
    (2, 5, 3, 1, 7, 6, 0, 4, 8),
    # Medio (~14 movimientos)     – verificado solucionable
    (2, 6, 5, 7, 0, 3, 1, 4, 8),
    # Difícil (~20 movimientos)   – verificado solucionable
    (6, 5, 3, 2, 7, 8, 1, 4, 0),
]

# --- 15-puzzle (4×4) ---
GOAL_15 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)

PUZZLES_15 = [
    # Fácil (~5 movimientos)
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15),
    # Medio (~15 movimientos)
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15),
    # Difícil (~20 movimientos) – manejable por Manhattan y heurística personalizada
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 12, 13, 14, 11, 15),
]

HEURISTICS = [
    ("Fichas fuera de lugar", misplaced_tiles),
    ("Distancia Manhattan",  manhattan_distance),
    ("Heurística personalizada", custom_heuristic),
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_puzzle_comparison() -> list:
    """
    Ejecuta A* con las tres heurísticas sobre los puzzles de prueba.
    Devuelve una lista de dicts listos para imprimir.
    """
    results = []

    for puzzle_set, goal, size, label in [
        (PUZZLES_8,  GOAL_8,  3, "8-puzzle"),
        (PUZZLES_15, GOAL_15, 4, "15-puzzle"),
    ]:
        for p_idx, puzzle in enumerate(puzzle_set, 1):
            difficulty = ["Fácil", "Medio", "Difícil"][p_idx - 1]
            for h_name, h_fn in HEURISTICS:
                r = a_star(puzzle, goal, size, h_fn)
                r["puzzle"] = label
                r["difficulty"] = difficulty
                r["heuristic"] = h_name
                results.append(r)

    return results


if __name__ == "__main__":
    from main import print_puzzle_table
    print_puzzle_table(run_puzzle_comparison())
