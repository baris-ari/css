"""
Schelling Segregation Model — minimal terminal implementation.

Date written: 2026-05-20
Author:       Claude Opus 4.7 (Anthropic), for teaching purposes.

This file is meant to be READ as much as RUN. Each section maps directly
to a numbered step in model_plan.md so you can follow along side-by-side.

Only the Python standard library is used (random, sys). Run with:
    python3 schelling.py
"""

# We use `random` to place agents randomly and to shuffle them later.
# `sys` is only used to read a line from stdin (the "press enter" prompt).
import random
import sys


# ---------------------------------------------------------------------------
# DEFAULT PARAMETERS  (model_plan.md, "Default Parameters")
# ---------------------------------------------------------------------------
# These are gathered in one place so a teacher/student can tweak them and
# immediately see the effect. They are passed into run() at the bottom.
N            = 20     # grid side length (the grid is N x N)
DENSITY      = 0.90   # fraction of cells that contain an agent (rest are empty)
SHARE_X      = 0.50   # of the agents, fraction that are type "X" (rest are "O")
TAU          = 0.30   # happiness threshold: similarity strictly < TAU => unhappy
MAX_ROUNDS   = 200    # safety cap on how many rounds we will simulate

# Symbols we use when drawing the grid in the terminal.
# Unhappy agents get a "bar" over them (x̄, ō) using a Unicode combining char.
EMPTY        = "."
X_HAPPY      = "X"
O_HAPPY      = "O"
COMBINING_BAR = "̄"            # combining macron, renders as the bar
X_UNHAPPY    = "x" + COMBINING_BAR  # x̄
O_UNHAPPY    = "o" + COMBINING_BAR  # ō


# ---------------------------------------------------------------------------
# STEP 1 + 2 — Build the grid and place agents randomly
# ---------------------------------------------------------------------------
def make_grid(n, density, share_x):
    """
    Return an n x n grid (a list of lists) filled with "X", "O", or EMPTY.

    We represent the grid as a list of lists of strings. grid[r][c] gives the
    cell at row r, column c. Rows go top-to-bottom, columns left-to-right.
    The grid has "hard edges": there is no wrap-around. Step 3 will handle
    the consequence (edge/corner cells simply have fewer neighbours).
    """
    total_cells   = n * n
    num_agents    = int(total_cells * density)        # how many cells get an agent
    num_x         = int(num_agents * share_x)         # of those, how many are "X"
    num_o         = num_agents - num_x                # the rest are "O"
    num_empty     = total_cells - num_agents          # remaining cells stay empty

    # Build a flat list of all cell contents, then shuffle it. This is a tiny
    # trick: instead of placing agents one-by-one and checking for collisions,
    # we make exactly the right multiset of contents and randomise the order.
    cells = [X_HAPPY] * num_x + [O_HAPPY] * num_o + [EMPTY] * num_empty
    random.shuffle(cells)

    # Reshape the flat list into n rows of n columns.
    grid = [cells[r * n : (r + 1) * n] for r in range(n)]
    return grid


# ---------------------------------------------------------------------------
# STEP 3 — Moore-neighbourhood similarity for a single agent
# ---------------------------------------------------------------------------
def neighbours(grid, r, c):
    """
    Yield the contents of every cell in the Moore neighbourhood of (r, c).

    Moore = the up-to-8 surrounding cells (4 orthogonal + 4 diagonal).
    Because the grid has hard edges, we skip any neighbour that would fall
    outside the grid. We also skip (r, c) itself — an agent is not its own
    neighbour.
    """
    n = len(grid)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue                            # skip the agent itself
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:         # stay inside the grid
                yield grid[nr][nc]


def similarity(grid, r, c):
    """
    Return the similarity ratio for the agent at (r, c).

    Definition (from the model_plan):
        similarity = (same-type neighbours) / (occupied neighbours)
    Empty cells do NOT count in the denominator. If the agent has zero
    occupied neighbours (isolated), we return None to signal "isolated".
    The caller treats isolated agents as happy and excludes them from the
    average-similarity statistic.
    """
    me = grid[r][c]
    same     = 0
    occupied = 0
    for cell in neighbours(grid, r, c):
        if cell == EMPTY:
            continue                                # empty cells are ignored
        occupied += 1
        if cell == me:
            same += 1
    if occupied == 0:
        return None                                 # isolated agent
    return same / occupied


# ---------------------------------------------------------------------------
# STEP 4 — Decide who is unhappy
# ---------------------------------------------------------------------------
def classify_agents(grid, tau):
    """
    Walk the whole grid once and return three lists of (row, col) tuples:
        unhappy   — agents whose similarity is strictly below tau
        happy     — agents who meet the threshold (or are isolated)
        empties   — empty cells (we need these later to relocate the unhappy)

    We also return the list of similarity values for non-isolated agents so
    that classify_agents + the statistics step share the same single pass.
    """
    unhappy, happy, empties = [], [], []
    sim_values = []                                 # for the average-similarity stat

    n = len(grid)
    for r in range(n):
        for c in range(n):
            cell = grid[r][c]
            if cell == EMPTY:
                empties.append((r, c))
                continue
            s = similarity(grid, r, c)
            if s is None:
                # Isolated agents are treated as happy per the model_plan.
                happy.append((r, c))
            else:
                sim_values.append(s)
                # The threshold uses strict less-than, as specified.
                if s < tau:
                    unhappy.append((r, c))
                else:
                    happy.append((r, c))

    return unhappy, happy, empties, sim_values


# ---------------------------------------------------------------------------
# STEP 5 — Round statistics
# ---------------------------------------------------------------------------
def statistics(unhappy, happy, sim_values):
    """
    Compute the two statistics required for each round:
      - the count and proportion of unhappy agents
      - the average similarity across non-isolated agents
    Returns a small dict so the print step can format it however it likes.
    """
    total_agents = len(unhappy) + len(happy)
    unhappy_prop = (len(unhappy) / total_agents) if total_agents else 0.0
    avg_sim      = (sum(sim_values) / len(sim_values)) if sim_values else 0.0
    return {
        "total":        total_agents,
        "unhappy":      len(unhappy),
        "unhappy_prop": unhappy_prop,
        "avg_sim":      avg_sim,
    }


# ---------------------------------------------------------------------------
# STEP 6 — Draw the grid (and statistics) to the terminal
# ---------------------------------------------------------------------------
def draw(grid, unhappy, stats, round_num, params):
    """
    Print the grid, with unhappy agents shown using x̄ / ō.

    We build a set of unhappy coordinates for O(1) membership tests, then
    walk the grid and choose the right symbol for each cell. The bottom of
    the output shows the round number, parameters, and statistics.
    """
    unhappy_set = set(unhappy)
    n = len(grid)

    # --- Title line ---
    print(
        f"Round {round_num}  |  n={params['n']}, density={params['density']}, "
        f"share_X={params['share_x']}, tau={params['tau']}"
    )

    # --- Grid ---
    for r in range(n):
        row_symbols = []
        for c in range(n):
            cell = grid[r][c]
            if cell == EMPTY:
                row_symbols.append(EMPTY)
            elif (r, c) in unhappy_set:
                # Replace the happy glyph with its unhappy (barred) variant.
                row_symbols.append(X_UNHAPPY if cell == X_HAPPY else O_UNHAPPY)
            else:
                row_symbols.append(cell)
        # A single space between cells keeps the grid roughly square in
        # most monospace fonts (because the bar adds visual width).
        print(" ".join(row_symbols))

    # --- Statistics line ---
    print(
        f"unhappy = {stats['unhappy']}/{stats['total']} "
        f"({stats['unhappy_prop']:.1%})   "
        f"avg similarity = {stats['avg_sim']:.3f}"
    )


# ---------------------------------------------------------------------------
# STEP 8 — Move unhappy agents into empty cells
# ---------------------------------------------------------------------------
def relocate(grid, unhappy, empties):
    """
    Move each unhappy agent to a randomly chosen empty cell.

    We shuffle both lists (per the model_plan) and then pair them up.
    Important subtlety: the number of unhappy agents may exceed the number
    of empty cells, so we only pair as many as the smaller list allows.
    The vacated cell of each moved agent itself becomes empty.
    """
    random.shuffle(unhappy)
    random.shuffle(empties)

    # Pair up unhappy agents with empty cells, up to the shorter list length.
    for (ur, uc), (er, ec) in zip(unhappy, empties):
        agent_type      = grid[ur][uc]              # remember what was there
        grid[er][ec]    = agent_type                # move agent into empty cell
        grid[ur][uc]    = EMPTY                     # old cell is now empty


# ---------------------------------------------------------------------------
# STEPS 7 + 9 — Main loop: simulate, print, ask to continue, repeat
# ---------------------------------------------------------------------------
def run(n=N, density=DENSITY, share_x=SHARE_X, tau=TAU, max_rounds=MAX_ROUNDS):
    """
    Drive the simulation. This function ties every step together.

    Loop structure (mirrors model_plan steps 3-9):
        for each round:
            classify agents              (steps 3, 4)
            compute statistics           (step 5)
            draw the grid                (step 6)
            if nobody is unhappy:  stop  (step 7)
            else: wait for user, move    (steps 7, 8)
    """
    # Bundle the params so draw() can print them in the title.
    params = {"n": n, "density": density, "share_x": share_x, "tau": tau}

    # Step 1+2 happen exactly once, before the loop starts.
    grid = make_grid(n, density, share_x)

    for round_num in range(1, max_rounds + 1):
        unhappy, happy, empties, sim_values = classify_agents(grid, tau)
        stats = statistics(unhappy, happy, sim_values)

        draw(grid, unhappy, stats, round_num, params)

        # Stopping rule: equilibrium reached, nothing left to do.
        if not unhappy:
            print("No unhappy agents remain — simulation finished.")
            return

        # Otherwise, give the user control over pacing, then move agents.
        # We use sys.stdin.readline so Ctrl-D / EOF cleanly ends the run
        # instead of raising EOFError as input() would.
        print("Press Enter for the next round (Ctrl-C to quit)...")
        if sys.stdin.readline() == "":
            print("Input closed — stopping.")
            return

        relocate(grid, unhappy, empties)

    # If we exit the loop normally, we hit the iteration cap.
    print(f"Reached max_rounds = {max_rounds} — stopping.")


# Standard Python idiom: only run the simulation when the file is executed
# directly (e.g. `python3 schelling.py`), not when it is imported.
if __name__ == "__main__":
    run()
