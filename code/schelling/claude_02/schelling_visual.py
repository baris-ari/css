"""
Schelling Segregation Model — visual version with matplotlib + numpy.

Date written: 2026-05-20
Author:       Claude Opus 4.7 (Anthropic), for teaching purposes.

This file is a richer companion to schelling.py. The model rules are
identical (see model_plan.md); only the implementation differs:

  - the grid is a numpy array (faster, and lets us compute every cell's
    similarity in one shot using array arithmetic),
  - matplotlib draws the grid in colour, with small white dots marking
    the agents who are about to move,
  - a side panel tracks the unhappy proportion and the average similarity
    over rounds, so the segregation dynamic is visible at a glance.

Run inside the project's virtual environment:
    .venv/bin/python schelling_visual.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# DEFAULT PARAMETERS  (same names as model_plan.md)
# ---------------------------------------------------------------------------
N            = 50      # grid side length (50 x 50 = 2500 cells — comfortable)
DENSITY      = 0.90    # fraction of cells that contain an agent
SHARE_X      = 0.50    # of the agents, fraction that are type X
TAU          = 0.30    # happiness threshold (strictly less => unhappy)
MAX_ROUNDS   = 200     # safety cap on iterations
PAUSE_SECS   = 0       # 0 = click/key to advance; >0 = auto-advance every N secs
SEED         = None    # set to an int (e.g. 42) for a reproducible run

# Integer codes for cell contents. Using ints makes numpy operations easy.
EMPTY, X, O = 0, 1, 2

# Colour for each cell code. The list order matches the codes 0/1/2 above.
#   empty  -> near-white          X -> blue            O -> orange
COLOURS = ["#FAFAFA", "#1F77B4", "#FF7F0E"]
CMAP    = ListedColormap(COLOURS)


# ---------------------------------------------------------------------------
# STEPS 1 + 2 — Build the grid and place agents randomly
# ---------------------------------------------------------------------------
def make_grid(n, density, share_x, rng):
    """
    Return an n x n numpy array whose values are EMPTY / X / O.

    Same shuffle-a-flat-list trick as the simple version: build a flat
    array with exactly the right counts of each value, shuffle it, then
    reshape into a square. This guarantees the densities are exact, with
    no collision-checking loops.
    """
    total       = n * n
    num_agents  = int(total * density)
    num_x       = int(num_agents * share_x)
    num_o       = num_agents - num_x
    num_empty   = total - num_agents

    flat = np.array([X] * num_x + [O] * num_o + [EMPTY] * num_empty,
                    dtype=np.int8)
    rng.shuffle(flat)
    return flat.reshape(n, n)


# ---------------------------------------------------------------------------
# STEP 3 — Moore-neighbourhood similarity, vectorised over the whole grid
# ---------------------------------------------------------------------------
def neighbour_sum(arr):
    """
    For every cell, sum its 8 Moore-neighbour values (excluding the cell
    itself). Works on any 2D numpy array.

    Trick: pad the array with a 1-cell border of zeros, then add up eight
    shifted copies of the padded array — one per neighbour direction.
    The zero border gives us the "hard edges" behaviour for free: cells
    on the edge get zero contributions from neighbours that don't exist.
    """
    n = arr.shape[0]
    padded = np.pad(arr, 1, mode="constant", constant_values=0)
    total  = np.zeros_like(arr)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue                                # skip the cell itself
            total += padded[1 + dr : 1 + dr + n,
                            1 + dc : 1 + dc + n]
    return total


def similarity_grid(grid):
    """
    Return two arrays the same shape as `grid`:
      sim    — same-type ratio per cell (NaN for empty cells and for
               isolated agents, so they are excluded from later averages)
      n_occ  — count of occupied neighbours per cell
    """
    # 0/1 masks for each cell type — these are the inputs to neighbour_sum.
    is_x   = (grid == X).astype(np.int16)
    is_o   = (grid == O).astype(np.int16)
    is_any = is_x + is_o                                # 1 if cell is occupied

    same_x = neighbour_sum(is_x)                        # X neighbours per cell
    same_o = neighbour_sum(is_o)                        # O neighbours per cell
    n_occ  = neighbour_sum(is_any)                      # any-type neighbours

    # "Same-as-me" count depends on what's at the cell.
    same = np.where(grid == X, same_x,
            np.where(grid == O, same_o, 0))

    # Divide where it makes sense; mark everything else NaN so averages skip it.
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where((grid != EMPTY) & (n_occ > 0),
                       same / np.maximum(n_occ, 1),
                       np.nan)
    return sim, n_occ


# ---------------------------------------------------------------------------
# STEPS 4 + 5 — Decide who is unhappy and compute round statistics
# ---------------------------------------------------------------------------
def round_state(grid, tau):
    """Bundle everything the round needs into a single dictionary."""
    sim, n_occ     = similarity_grid(grid)

    occupied_mask  = (grid != EMPTY)
    # An agent is unhappy iff it has at least one neighbour AND sim < tau.
    # Isolated agents (n_occ == 0) are treated as happy per the plan.
    unhappy_mask   = occupied_mask & (n_occ > 0) & (sim < tau)

    total_agents   = int(occupied_mask.sum())
    n_unhappy      = int(unhappy_mask.sum())
    # nanmean ignores NaNs, which is exactly the "exclude isolated agents" rule.
    avg_sim        = float(np.nanmean(sim)) if total_agents else 0.0

    return {
        "unhappy_mask":   unhappy_mask,
        "total_agents":   total_agents,
        "n_unhappy":      n_unhappy,
        "unhappy_prop":   (n_unhappy / total_agents) if total_agents else 0.0,
        "avg_sim":        avg_sim,
    }


# ---------------------------------------------------------------------------
# STEP 8 — Move unhappy agents to random empty cells
# ---------------------------------------------------------------------------
def relocate(grid, unhappy_mask, rng):
    """
    Pair shuffled unhappy positions with shuffled empty positions, then
    move agent-by-agent. If unhappy agents outnumber empty cells, only as
    many can move as there are empties (the rest stay put for this round).
    """
    unhappy_idx = np.argwhere(unhappy_mask)             # array of [row, col]
    empty_idx   = np.argwhere(grid == EMPTY)

    rng.shuffle(unhappy_idx)
    rng.shuffle(empty_idx)

    k = min(len(unhappy_idx), len(empty_idx))
    for i in range(k):
        ur, uc = unhappy_idx[i]
        er, ec = empty_idx[i]
        grid[er, ec] = grid[ur, uc]                     # agent moves in
        grid[ur, uc] = EMPTY                            # old cell now empty


# ---------------------------------------------------------------------------
# STEP 6 — Build the figure once, then mutate the same artists each round.
# ---------------------------------------------------------------------------
def setup_figure(n, params):
    """
    Create the figure and return references to the artists we will update
    each round. Building everything once and then mutating it (instead of
    redrawing from scratch) is what keeps the animation smooth.

    Layout:
      ┌─────────────┬──────────────┐
      │   grid      │  stats over  │
      │ (imshow +   │   time       │
      │  unhappy    │  (two lines) │
      │  markers)   │              │
      └─────────────┴──────────────┘
    """
    fig, (ax_grid, ax_stats) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    fig.canvas.manager.set_window_title("Schelling Segregation Model")

    # --- Grid axes ---------------------------------------------------------
    # vmin/vmax pin the colour scale so 0/1/2 always map to empty/X/O.
    grid_img = ax_grid.imshow(np.zeros((n, n), dtype=np.int8),
                              cmap=CMAP, vmin=0, vmax=2,
                              interpolation="nearest")
    # A scatter overlay highlights agents who are about to move.
    # We start empty; update_figure rewrites the offsets each round.
    unhappy_dots = ax_grid.scatter([], [], s=18, marker="o",
                                   facecolors="white", edgecolors="black",
                                   linewidths=0.6, zorder=3)
    ax_grid.set_xticks([]); ax_grid.set_yticks([])
    # Thin white grid lines between cells — only for small N, else too noisy.
    if n <= 30:
        ax_grid.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax_grid.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax_grid.grid(which="minor", color="white", linewidth=0.5)
        ax_grid.tick_params(which="minor", length=0)

    title      = ax_grid.set_title("", fontsize=11)
    stats_text = ax_grid.text(0.5, -0.04, "",
                              transform=ax_grid.transAxes,
                              ha="center", va="top", fontsize=10)

    # --- Stats axes --------------------------------------------------------
    ax_stats.set_xlim(0, 20)
    ax_stats.set_ylim(0, 1)
    ax_stats.set_xlabel("round")
    ax_stats.set_ylabel("value (0 – 1)")
    ax_stats.set_title(
        f"n={params['n']}, density={params['density']}, "
        f"share_X={params['share_x']}, τ={params['tau']}",
        fontsize=11,
    )
    (line_unhappy,) = ax_stats.plot([], [], color="#D62728",
                                    label="unhappy proportion")
    (line_sim,)     = ax_stats.plot([], [], color="#2CA02C",
                                    label="average similarity")
    ax_stats.axhline(params["tau"], color="grey", linestyle=":",
                     linewidth=1, label=f"τ = {params['tau']}")
    ax_stats.legend(loc="lower right", fontsize=9)
    ax_stats.grid(True, alpha=0.3)

    fig.tight_layout()
    return {
        "fig":          fig,
        "grid_img":     grid_img,
        "unhappy_dots": unhappy_dots,
        "title":        title,
        "stats_text":   stats_text,
        "ax_stats":     ax_stats,
        "line_unhappy": line_unhappy,
        "line_sim":     line_sim,
    }


def update_figure(art, grid, state, round_num, history):
    """Push the latest grid + statistics into the existing matplotlib artists."""
    # The grid colours: just hand imshow a new 2D array.
    art["grid_img"].set_data(grid)

    # Unhappy markers — scatter expects (x, y) where x = col, y = row.
    ur, uc = np.where(state["unhappy_mask"])
    offsets = np.column_stack([uc, ur]) if len(ur) else np.empty((0, 2))
    art["unhappy_dots"].set_offsets(offsets)

    art["title"].set_text(f"Round {round_num}")
    art["stats_text"].set_text(
        f"unhappy = {state['n_unhappy']}/{state['total_agents']} "
        f"({state['unhappy_prop']:.1%})    "
        f"average similarity = {state['avg_sim']:.3f}"
    )

    # History lines — overwrite both x-data and y-data each round.
    art["line_unhappy"].set_data(history["round"], history["unhappy"])
    art["line_sim"].set_data(history["round"],     history["avg_sim"])
    art["ax_stats"].set_xlim(0, max(20, round_num + 1))

    art["fig"].canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main loop  (steps 7 + 8 + 9)
# ---------------------------------------------------------------------------
def run(n=N, density=DENSITY, share_x=SHARE_X, tau=TAU,
        max_rounds=MAX_ROUNDS, pause=PAUSE_SECS, seed=SEED):
    """Drive the simulation and refresh the figure once per round."""
    rng     = np.random.default_rng(seed)
    grid    = make_grid(n, density, share_x, rng)
    params  = {"n": n, "density": density, "share_x": share_x, "tau": tau}

    plt.ion()                                           # interactive: live updates
    art     = setup_figure(n, params)
    history = {"round": [], "unhappy": [], "avg_sim": []}

    for round_num in range(1, max_rounds + 1):
        state = round_state(grid, tau)

        history["round"].append(round_num)
        history["unhappy"].append(state["unhappy_prop"])
        history["avg_sim"].append(state["avg_sim"])

        update_figure(art, grid, state, round_num, history)

        # Step 7: stop if nobody is unhappy.
        if state["n_unhappy"] == 0:
            art["title"].set_text(f"Round {round_num} — equilibrium reached")
            art["fig"].canvas.draw_idle()
            break

        # Pacing. If the user has closed the window, stop gracefully.
        if not plt.fignum_exists(art["fig"].number):
            return
        if pause > 0:
            plt.pause(pause)
        else:
            # Show a hint while we wait for input, then clear it once we move on.
            art["title"].set_text(
                f"Round {round_num}  —  click or press any key for next round"
            )
            art["fig"].canvas.draw_idle()
            plt.waitforbuttonpress()

        # Step 8: relocate the unhappy agents.
        relocate(grid, state["unhappy_mask"], rng)
    else:
        # Python detail: a for-loop's `else` runs only if we did NOT break.
        # Reaching here means we hit max_rounds without equilibrium.
        art["title"].set_text(f"Stopped at max_rounds = {max_rounds}")
        art["fig"].canvas.draw_idle()

    plt.ioff()
    plt.show()                                          # keep window open till closed


if __name__ == "__main__":
    run()
