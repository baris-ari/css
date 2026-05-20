"""
Schelling Segregation Model

Date written: 2026-05-20
Written by: Codex, an AI coding agent based on GPT-5

Purpose:
    This is a minimal, terminal-based implementation of Schelling's
    segregation model. The code follows model_plan.md directly.

    The comments explain the Python implementation choices, not the social
    science model itself.
"""

import random


# ---------------------------------------------------------------------------
# Default parameters from model_plan.md
# ---------------------------------------------------------------------------

N = 20                  # Grid width and height. The grid has N * N cells.
DENSITY = 0.90          # Fraction of cells that start with an agent.
SHARE_X = 0.50          # Fraction of agents that are type X.
THRESHOLD = 0.30        # Tau: minimum similarity needed to be happy.
MAX_ROUNDS = 200        # Safety cap so the model cannot run forever.


# We store each cell as one simple string.
# "." means empty. "X" and "O" mean occupied by one of the two agent types.
EMPTY = "."
TYPE_X = "X"
TYPE_O = "O"


# ---------------------------------------------------------------------------
# 1. Create a square grid of n x n
# ---------------------------------------------------------------------------

def make_empty_grid(n):
    """Return an n by n grid filled with empty cells."""

    # A grid is a list of rows.
    # Each row is a list of cell values.
    return [[EMPTY for _ in range(n)] for _ in range(n)]


# ---------------------------------------------------------------------------
# 2. Create and place agents randomly
# ---------------------------------------------------------------------------

def create_agents(n, density, share_x):
    """Create a list of agents, using the requested density and X share."""

    total_cells = n * n
    agent_count = round(total_cells * density)

    # Decide how many agents are X. The remaining agents are O.
    x_count = round(agent_count * share_x)
    o_count = agent_count - x_count

    # The agents are represented only by their type.
    return [TYPE_X] * x_count + [TYPE_O] * o_count


def place_agents_randomly(grid, agents):
    """Put agents into random empty cells on the grid."""

    n = len(grid)

    # First list every possible position in the grid.
    all_positions = []
    for row in range(n):
        for col in range(n):
            all_positions.append((row, col))

    # Shuffle positions and agents so placement is random.
    random.shuffle(all_positions)
    random.shuffle(agents)

    # Pair each agent with one position and write the agent into that cell.
    for agent, (row, col) in zip(agents, all_positions):
        grid[row][col] = agent


def make_initial_grid(n, density, share_x):
    """Build the starting grid for the model."""

    grid = make_empty_grid(n)
    agents = create_agents(n, density, share_x)
    place_agents_randomly(grid, agents)
    return grid


# ---------------------------------------------------------------------------
# 3. Calculate neighbour similarity
# ---------------------------------------------------------------------------

def neighbour_positions(row, col, n):
    """Return valid Moore-neighbour positions around one cell."""

    positions = []

    # dr and dc mean "difference in row" and "difference in column".
    # The values -1, 0, and 1 cover the surrounding 3 by 3 square.
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            # Skip the centre cell. A cell is not its own neighbour.
            if dr == 0 and dc == 0:
                continue

            neighbour_row = row + dr
            neighbour_col = col + dc

            # Hard edges: only keep positions that are inside the grid.
            if 0 <= neighbour_row < n and 0 <= neighbour_col < n:
                positions.append((neighbour_row, neighbour_col))

    return positions


def calculate_similarity(grid, row, col):
    """
    Calculate same-type neighbours divided by occupied neighbours.

    Empty neighbouring cells are ignored. If an agent has no occupied
    neighbours, return None so the statistics code can exclude it from the
    average. The decision-rule code treats None as happy.
    """

    agent = grid[row][col]
    n = len(grid)
    same_type_neighbours = 0
    occupied_neighbours = 0

    for neighbour_row, neighbour_col in neighbour_positions(row, col, n):
        neighbour = grid[neighbour_row][neighbour_col]

        # Empty cells do not count in the denominator.
        if neighbour == EMPTY:
            continue

        occupied_neighbours += 1

        if neighbour == agent:
            same_type_neighbours += 1

    if occupied_neighbours == 0:
        return None

    return same_type_neighbours / occupied_neighbours


# ---------------------------------------------------------------------------
# 4. Is the similarity below a threshold?
# ---------------------------------------------------------------------------

def is_unhappy(similarity, threshold):
    """Return True if an agent is unhappy."""

    # Isolated agents have similarity None and are treated as happy.
    if similarity is None:
        return False

    # The plan says unhappy means strictly below the threshold.
    return similarity < threshold


def find_unhappy_agents(grid, threshold):
    """Return a list of positions for all unhappy agents."""

    unhappy_agents = []
    n = len(grid)

    for row in range(n):
        for col in range(n):
            if grid[row][col] == EMPTY:
                continue

            similarity = calculate_similarity(grid, row, col)
            if is_unhappy(similarity, threshold):
                unhappy_agents.append((row, col))

    return unhappy_agents


# ---------------------------------------------------------------------------
# 5. Calculate relevant statistics
# ---------------------------------------------------------------------------

def calculate_statistics(grid, threshold):
    """Count unhappy agents and calculate average similarity."""

    agent_count = 0
    unhappy_count = 0
    similarities_for_average = []
    n = len(grid)

    for row in range(n):
        for col in range(n):
            if grid[row][col] == EMPTY:
                continue

            agent_count += 1
            similarity = calculate_similarity(grid, row, col)

            if is_unhappy(similarity, threshold):
                unhappy_count += 1

            # Isolated agents are disregarded for average similarity.
            if similarity is not None:
                similarities_for_average.append(similarity)

    if similarities_for_average:
        average_similarity = sum(similarities_for_average) / len(similarities_for_average)
    else:
        average_similarity = None

    unhappy_proportion = unhappy_count / agent_count

    return {
        "agent_count": agent_count,
        "unhappy_count": unhappy_count,
        "unhappy_proportion": unhappy_proportion,
        "average_similarity": average_similarity,
    }


# ---------------------------------------------------------------------------
# 6. Plot the grid
# ---------------------------------------------------------------------------

def display_cell(grid, row, col, unhappy_positions):
    """Return the terminal character to show for one cell."""

    cell = grid[row][col]

    if cell == EMPTY:
        return EMPTY

    # The plan asks us to show unhappy agents with a bar: x̄ and ō.
    if (row, col) in unhappy_positions:
        if cell == TYPE_X:
            return "x̄"
        return "ō"

    return cell


def plot_grid(grid, unhappy_agents, statistics, round_number, threshold, density, share_x):
    """Print the grid and round statistics to the terminal."""

    unhappy_positions = set(unhappy_agents)

    print()
    print(
        f"Round {round_number} | tau={threshold:.2f} | "
        f"n={len(grid)} | density={density:.2f} | share_X={share_x:.2f}"
    )
    print("-" * (len(grid) * 3))

    for row in range(len(grid)):
        displayed_row = []
        for col in range(len(grid)):
            displayed_row.append(display_cell(grid, row, col, unhappy_positions))
        print(" ".join(displayed_row))

    print("-" * (len(grid) * 3))
    print(f"Unhappy agents: {statistics['unhappy_count']} of {statistics['agent_count']}")
    print(f"Unhappy proportion: {statistics['unhappy_proportion']:.2f}")

    if statistics["average_similarity"] is None:
        print("Average similarity: not available")
    else:
        print(f"Average similarity: {statistics['average_similarity']:.2f}")


# ---------------------------------------------------------------------------
# 8. Move unhappy agents to an empty cell randomly
# ---------------------------------------------------------------------------

def find_empty_cells(grid):
    """Return a list of positions for all empty cells."""

    empty_cells = []
    n = len(grid)

    for row in range(n):
        for col in range(n):
            if grid[row][col] == EMPTY:
                empty_cells.append((row, col))

    return empty_cells


def move_unhappy_agents(grid, unhappy_agents):
    """Move each unhappy agent to a randomly chosen empty cell."""

    empty_cells = find_empty_cells(grid)

    # Shuffle both lists, as requested in the plan.
    random.shuffle(unhappy_agents)
    random.shuffle(empty_cells)

    # Move as many unhappy agents as there are empty cells.
    # With the default density there should be enough empty cells.
    for old_position, new_position in zip(unhappy_agents, empty_cells):
        old_row, old_col = old_position
        new_row, new_col = new_position

        grid[new_row][new_col] = grid[old_row][old_col]
        grid[old_row][old_col] = EMPTY


# ---------------------------------------------------------------------------
# 7 and 9. Stop or continue, and repeat the loop
# ---------------------------------------------------------------------------

def run_model(n=N, density=DENSITY, share_x=SHARE_X, threshold=THRESHOLD, max_rounds=MAX_ROUNDS):
    """Run the model until everyone is happy or max_rounds is reached."""

    grid = make_initial_grid(n, density, share_x)

    for round_number in range(max_rounds + 1):
        unhappy_agents = find_unhappy_agents(grid, threshold)
        statistics = calculate_statistics(grid, threshold)

        plot_grid(grid, unhappy_agents, statistics, round_number, threshold, density, share_x)

        if not unhappy_agents:
            print("No unhappy agents remain. Stopping.")
            break

        if round_number == max_rounds:
            print("Maximum number of rounds reached. Stopping.")
            break

        input("Press Enter to move unhappy agents to the next round...")
        move_unhappy_agents(grid, unhappy_agents)


if __name__ == "__main__":
    run_model()
