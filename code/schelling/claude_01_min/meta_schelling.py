# meta_schelling.py
# =================
# Meta-simulation: how does the similarity threshold affect final segregation?
#
# We sweep the threshold from 0.20 to 0.60 in steps of 0.025 (17 values).
# At each threshold we run N_SIMULATIONS independent Schelling simulations,
# let each one run to convergence, then record the final "percent similar".
# We plot the mean percent-similar against the threshold.
#
# "Percent similar" is the NetLogo metric: for every agent, compute the
# fraction of its occupied neighbours who share its type, then average
# across all agents and multiply by 100.
#
# Requires: matplotlib  (pip install matplotlib)
# Everything else is base Python.

import random
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

GRID_SIZE      = 20      # grid is GRID_SIZE × GRID_SIZE cells
RATIO_X        = 0.45    # fraction of cells that start as X agents
RATIO_O        = 0.45    # fraction of cells that start as O agents
                          # empty fraction = 1 − RATIO_X − RATIO_O = 0.20

N_SIMULATIONS  = 300     # how many independent runs per threshold value
MAX_ROUNDS     = 500     # give up on a single run after this many rounds
                          # (prevents infinite loops on very high thresholds)

THRESHOLD_MIN  = 0.10    # lowest threshold to test
THRESHOLD_MAX  = 0.60    # highest threshold to test
THRESHOLD_STEP = 0.01    # step between thresholds  → 17 values total


# ─────────────────────────────────────────────────────────────────────────────
#  CORE MODEL — same logic as schelling.py, stripped of all display code.
#  The threshold is now a parameter rather than a global constant so we can
#  pass different values in each simulation.
# ─────────────────────────────────────────────────────────────────────────────

def create_grid():
    """Place X agents, O agents, and empty cells randomly on the grid."""
    total  = GRID_SIZE * GRID_SIZE
    n_x    = int(total * RATIO_X)
    n_o    = int(total * RATIO_O)
    n_empty = total - n_x - n_o

    flat = ['X'] * n_x + ['O'] * n_o + ['.'] * n_empty
    random.shuffle(flat)

    # Reshape flat list into a 2-D list of rows.
    return [flat[r * GRID_SIZE : (r + 1) * GRID_SIZE] for r in range(GRID_SIZE)]


def get_neighbors(grid, row, col):
    """Return the values of the (up to 8) Moore-neighbourhood cells."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                neighbors.append(grid[r][c])
    return neighbors


def is_happy(grid, row, col, threshold):
    """Return True if the agent at (row, col) meets the similarity threshold."""
    agent = grid[row][col]
    if agent == '.':
        return True

    neighbors = get_neighbors(grid, row, col)
    occupied  = [n for n in neighbors if n != '.']

    if not occupied:
        return True   # no neighbours — nothing to be unhappy about

    same = sum(1 for n in occupied if n == agent)
    return (same / len(occupied)) >= threshold


def find_unhappy_agents(grid, threshold):
    """Return a list of (row, col) for every unhappy agent."""
    return [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if grid[r][c] != '.' and not is_happy(grid, r, c, threshold)
    ]


def move_agents(grid, unhappy_agents):
    """Move each unhappy agent to a randomly chosen empty cell."""
    # Collect all vacancies once, then update the list as agents move.
    empty_cells = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if grid[r][c] == '.'
    ]
    random.shuffle(unhappy_agents)

    for (r, c) in unhappy_agents:
        if not empty_cells:
            break
        idx       = random.randrange(len(empty_cells))
        tr, tc    = empty_cells[idx]
        grid[tr][tc] = grid[r][c]
        grid[r][c]   = '.'
        empty_cells[idx] = (r, c)   # old cell is now the new vacancy


# ─────────────────────────────────────────────────────────────────────────────
#  MEASUREMENT — the NetLogo "percent similar" metric
# ─────────────────────────────────────────────────────────────────────────────

def percent_similar(grid):
    """Compute the NetLogo percent-similar metric on the current grid.

    For each agent, find the fraction of its occupied neighbours who share
    its type.  Average those fractions across all agents and multiply by 100.

    This is identical to NetLogo's built-in measure so results are comparable.
    Returns a value between 0 (fully mixed) and 100 (fully segregated).
    """
    scores = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == '.':
                continue
            agent    = grid[r][c]
            neighbors = get_neighbors(grid, r, c)
            occupied  = [n for n in neighbors if n != '.']
            if occupied:
                same = sum(1 for n in occupied if n == agent)
                scores.append(same / len(occupied))

    if not scores:
        return 0.0
    return (sum(scores) / len(scores)) * 100   # expressed as a percentage


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_one_simulation(threshold):
    """Run one Schelling simulation silently.  Return final percent-similar.

    The simulation advances until all agents are happy or MAX_ROUNDS is
    reached.  We record the state at that stopping point — whether the model
    converged or hit the cap.
    """
    grid = create_grid()

    for _ in range(MAX_ROUNDS):
        unhappy = find_unhappy_agents(grid, threshold)
        if not unhappy:
            break                    # everyone is happy — converged
        move_agents(grid, unhappy)

    return percent_similar(grid)


# ─────────────────────────────────────────────────────────────────────────────
#  META-SIMULATION SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def build_threshold_list():
    """Build the list of 17 threshold values from 0.20 to 0.60.

    We use rounding to avoid floating-point drift
    (e.g. 0.20000000000000004 instead of 0.20).
    """
    n_steps = round((THRESHOLD_MAX - THRESHOLD_MIN) / THRESHOLD_STEP)
    return [round(THRESHOLD_MIN + i * THRESHOLD_STEP, 4) for i in range(n_steps + 1)]


def run_sweep():
    """Run N_SIMULATIONS at each threshold.  Return (thresholds, means).

    Prints a progress line for each threshold so the terminal isn't silent
    during what can be several thousand simulations.
    """
    thresholds = build_threshold_list()
    means      = []

    total_sims = len(thresholds) * N_SIMULATIONS
    print(f"Running {len(thresholds)} thresholds × {N_SIMULATIONS} simulations "
          f"= {total_sims} total runs …\n")

    for i, threshold in enumerate(thresholds):
        results = [run_one_simulation(threshold) for _ in range(N_SIMULATIONS)]
        mean_ps = sum(results) / len(results)
        means.append(mean_ps)

        # Progress update — one line per threshold so you can watch it build.
        print(f"  threshold {threshold:.3f}  →  mean percent-similar = {mean_ps:.1f}%"
              f"  [{i + 1}/{len(thresholds)}]")

    print()
    return thresholds, means


# ─────────────────────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(thresholds, means):
    """Draw a clean line plot: threshold on x-axis, mean percent-similar on y."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, means, marker='o', linewidth=2,
            markersize=6, color='steelblue')

    # Mark each data point with its value so the numbers are readable without
    # hovering — useful when showing this in class.
    for x, y in zip(thresholds, means):
        ax.annotate(f'{y:.1f}',
                    xy=(x, y), xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', fontsize=7, color='steelblue')

    ax.set_xlabel('Similarity threshold', fontsize=12)
    ax.set_ylabel('Mean percent similar (%)', fontsize=12)
    ax.set_title(
        f'Schelling model: threshold vs. segregation\n'
        f'({N_SIMULATIONS} simulations per point, '
        f'{GRID_SIZE}×{GRID_SIZE} grid, '
        f'{int((RATIO_X + RATIO_O) * 100)}% occupied)',
        fontsize=11
    )

    # Add a faint horizontal reference line at 50% — the expected value for
    # a perfectly random, fully mixed grid — so the "lift" from segregation
    # is visually obvious.
    ax.axhline(50, linestyle='--', color='grey', linewidth=0.8, label='random baseline (50%)')
    ax.legend(fontsize=9)

    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_xticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    

    plt.tight_layout()
    plt.savefig('meta_schelling.png', dpi=150)
    print("  Plot saved to meta_schelling.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    thresholds, means = run_sweep()
    plot_results(thresholds, means)