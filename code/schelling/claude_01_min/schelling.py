# schelling.py
# ============
# Schelling's Segregation Model — a minimal teaching implementation.
#
# The big idea: even agents with mild preferences for similar neighbours
# (e.g. "I just want 30% of my neighbours to be like me") will end up
# clustering into heavily segregated groups over many moves.
# Thomas Schelling introduced this model in 1971.
#
# How it works:
#   1. Place two groups (X and O) randomly on a grid, with some empty cells.
#   2. Each round, find every agent who is "unhappy" (too few similar neighbours).
#   3. Move each unhappy agent to a random empty cell.
#   4. Repeat until everyone is happy, or we reach the round limit.

import random


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS — change these to explore the model
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE            = 20    # the grid will be GRID_SIZE × GRID_SIZE cells
RATIO_X              = 0.40  # fraction of all cells that start as X agents
RATIO_O              = 0.40  # fraction of all cells that start as O agents
                              # remaining cells are empty: 1 − RATIO_X − RATIO_O
SIMILARITY_THRESHOLD = 0.30  # minimum fraction of same-type neighbours for happiness
                              # e.g. 0.30 means "at least 30% of my neighbours
                              # must be like me, or I want to move"
MAX_ROUNDS           = 100    # stop even if agents are still unhappy after this many rounds


# ─────────────────────────────────────────────────────────────────────────────
#  GRID CREATION
# ─────────────────────────────────────────────────────────────────────────────

def create_grid():
    """Build the starting grid with agents placed at random positions.

    The grid is a list of rows; each row is a list of cells.
    Each cell holds one of three values:
        'X'  — an X-group agent
        'O'  — an O-group agent
        '.'  — an empty cell (a vacancy an unhappy agent can move into)
    """
    total_cells = GRID_SIZE * GRID_SIZE

    n_x     = int(total_cells * RATIO_X)          # how many X agents
    n_o     = int(total_cells * RATIO_O)           # how many O agents
    n_empty = total_cells - n_x - n_o             # everything else is empty

    # Build a flat list with the right counts, shuffle it, then reshape into rows.
    flat = ['X'] * n_x + ['O'] * n_o + ['.'] * n_empty
    random.shuffle(flat)

    # Split the flat list into GRID_SIZE rows of GRID_SIZE cells each.
    grid = []
    for row in range(GRID_SIZE):
        start = row * GRID_SIZE
        grid.append(flat[start : start + GRID_SIZE])

    return grid


# ─────────────────────────────────────────────────────────────────────────────
#  NEIGHBOURHOOD LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_neighbors(grid, row, col):
    """Return the values of all occupied cells surrounding (row, col).

    We use the Moore neighbourhood: the 8 cells that share a side or corner
    with the centre cell.  Cells at the grid edge simply have fewer neighbours
    (we do NOT wrap around — there is no torus).
    Empty cells ('.') are included in the list so the caller can decide
    whether to count them or not.
    """
    neighbors = []
    for dr in [-1, 0, 1]:        # row offsets: up, same, down
        for dc in [-1, 0, 1]:    # column offsets: left, same, right
            if dr == 0 and dc == 0:
                continue          # skip the cell itself
            r, c = row + dr, col + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                neighbors.append(grid[r][c])
    return neighbors


def is_happy(grid, row, col):
    """Return True if the agent at (row, col) is satisfied with its neighbourhood.

    An agent is happy when the fraction of its *occupied* neighbours who
    share its type is at least SIMILARITY_THRESHOLD.

    Special cases:
    - Empty cells are always considered happy (they are not agents).
    - An agent with no occupied neighbours at all is considered happy
      (it has nothing to be unhappy about — no one is different from it).
    """
    agent = grid[row][col]
    if agent == '.':
        return True

    neighbors = get_neighbors(grid, row, col)
    occupied  = [n for n in neighbors if n != '.']  # ignore empty cells

    if not occupied:
        return True   # isolated agent — no neighbours to dislike

    same_type         = sum(1 for n in occupied if n == agent)
    fraction_similar  = same_type / len(occupied)

    return fraction_similar >= SIMILARITY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  FINDING AND MOVING UNHAPPY AGENTS
# ─────────────────────────────────────────────────────────────────────────────

def find_unhappy_agents(grid):
    """Return a list of (row, col) positions for every unhappy agent."""
    unhappy = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] != '.' and not is_happy(grid, r, c):
                unhappy.append((r, c))
    return unhappy


def find_empty_cells(grid):
    """Return a list of (row, col) positions for every empty cell."""
    return [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if grid[r][c] == '.'
    ]


def move_agents(grid, unhappy_agents):
    """Relocate each unhappy agent to a randomly chosen empty cell.

    This is the heart of the Schelling model.  Unhappy agents move; happy
    ones stay put.  Agents move to a *random* vacancy — not necessarily one
    that will make them happy.  Over many rounds, segregated clusters still
    emerge, which is the surprising result.

    We process unhappy agents in random order so no group gets priority,
    and we update the vacancy list on the fly so two agents don't land in
    the same cell.
    """
    empty_cells = find_empty_cells(grid)
    random.shuffle(unhappy_agents)   # random move order — no favouritism

    for (r, c) in unhappy_agents:
        if not empty_cells:
            break   # grid is full — no vacancies left (unlikely but safe)

        # Pick a random vacancy.
        idx       = random.randrange(len(empty_cells))
        tr, tc    = empty_cells[idx]

        # Perform the move.
        grid[tr][tc] = grid[r][c]   # place the agent in the new cell
        grid[r][c]   = '.'          # the old cell is now empty

        # Keep the vacancy list consistent:
        # the old cell (r, c) is now empty, so swap it into the list
        # in place of the target cell we just used.
        empty_cells[idx] = (r, c)


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_grid(grid, unhappy_set):
    """Print the grid using two-character symbols so columns stay aligned.

    Symbol key:
        'X ' — happy X agent
        'O ' — happy O agent
        '. ' — empty cell
        'X° ' — unhappy X agent  (wants to move)
        'O° ' — unhappy O agent  (wants to move)
    """
    for r in range(GRID_SIZE):
        row_str = ''
        for c in range(GRID_SIZE):
            agent = grid[r][c]
            if (r, c) in unhappy_set:
                row_str += agent + '° '  # combining asterisk above — sits on top of the letter
            elif agent == '.':
                row_str += '. '
            else:
                row_str += agent + ' '  # trailing space keeps columns even
        print(row_str)


def compute_segregation_index(grid):
    """Return a simple measure of how segregated the grid is.

    For each agent, compute the fraction of its occupied neighbours who
    share its type.  Average this across all agents.

    Interpretation:
        0.0  — every agent is completely surrounded by the other type
        1.0  — every agent is completely surrounded by its own type
        ~0.5 — fully random / integrated starting point
    """
    scores = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == '.':
                continue
            agent     = grid[r][c]
            neighbors = get_neighbors(grid, r, c)
            occupied  = [n for n in neighbors if n != '.']
            if occupied:
                same = sum(1 for n in occupied if n == agent)
                scores.append(same / len(occupied))

    return sum(scores) / len(scores) if scores else 0.0


def print_stats(round_num, n_agents, n_unhappy, grid):
    """Print a one-round summary: who moved and how segregated things are."""
    pct_unhappy = (n_unhappy / n_agents * 100) if n_agents > 0 else 0
    seg_index   = compute_segregation_index(grid)

    print()
    print(f"  Round            : {round_num}")
    print(f"  Unhappy agents   : {n_unhappy} of {n_agents}  ({pct_unhappy:.1f}%)")
    print(f"  Segregation index: {seg_index:.2f}  "
          "(0 = fully mixed  →  1 = fully segregated)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation():
    """Run the model round by round, pausing for the user between each round.

    Each round:
        1. Find all unhappy agents.
        2. Print the grid (marking unhappy agents with *).
        3. Print round statistics.
        4. If everyone is happy, stop.
        5. Wait for the user to press Enter.
        6. Move unhappy agents to random vacancies.
        7. Go to step 1.
    """
    grid = create_grid()

    # Count total agents once — this never changes (agents move, not appear/disappear).
    n_agents = sum(
        1 for r in range(GRID_SIZE)
          for c in range(GRID_SIZE)
          if grid[r][c] != '.'
    )

    # ── Welcome banner ────────────────────────────────────────────────────────
    width = GRID_SIZE * 2
    print("=" * width)
    print("  Schelling Segregation Model")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE}   "
          f"Threshold: {SIMILARITY_THRESHOLD:.0%}   "
          f"Agents: {n_agents}")
    print("=" * width)
    print("  X  happy X agent     O  happy O agent     .  empty cell")
    print("  X°  unhappy X agent   O°  unhappy O agent")
    print("=" * width)
    print()
    input("  Press Enter to start …")
    print()

    # ── Round loop ────────────────────────────────────────────────────────────
    for round_num in range(1, MAX_ROUNDS + 1):

        # Find who is unhappy this round.
        unhappy_agents = find_unhappy_agents(grid)
        unhappy_set    = set(unhappy_agents)   # set for fast (r,c) lookup in print_grid
        n_unhappy      = len(unhappy_agents)

        # Show current state.
        print_grid(grid, unhappy_set)
        print_stats(round_num, n_agents, n_unhappy, grid)

        # Check stopping conditions.
        if n_unhappy == 0:
            print("  ✓ All agents are happy — simulation complete.")
            break

        if round_num == MAX_ROUNDS:
            print(f"  Reached the {MAX_ROUNDS}-round limit. Stopping.")
            break

        # Pause, then advance.
        input("  Press Enter for the next round …")
        print()
        move_agents(grid, unhappy_agents)

    print()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_simulation()
