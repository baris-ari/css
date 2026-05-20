# Schelling Segregation Model

A minimal, heavily-commented Python implementation of Thomas Schelling's
segregation model for teaching purposes.

## What the model shows

Place two groups of agents (X and O) randomly on a grid, with some empty
cells. Each round, any agent who has fewer same-type neighbours than a
threshold picks up and moves to a random empty cell. Even with a very mild
threshold (e.g. 30%), strongly segregated clusters emerge over time. This
is Schelling's surprising result: macro-level segregation does not require
micro-level prejudice.

## Setup

You need Python 3.8 or later. No third-party packages are required.

```bash
# 1. Create a virtual environment inside the project folder
python3 -m venv venv

# 2. Activate it
#    On macOS / Linux:
source venv/bin/activate
#    On Windows:
venv\Scripts\activate

# 3. Run the simulation
python schelling.py
```

To leave the virtual environment when you're done:

```bash
deactivate
```

## How to use it

- The simulation runs round by round in the terminal.
- After each round you see the grid and a short stat summary.
- Press **Enter** to advance to the next round.
- It stops automatically when all agents are happy, or after `MAX_ROUNDS`.

## Tweaking the parameters

Open `schelling.py` and edit the variables at the top of the file:

| Variable              | Default | Meaning                                              |
|-----------------------|---------|------------------------------------------------------|
| `GRID_SIZE`           | 20      | Grid is `GRID_SIZE × GRID_SIZE` cells                |
| `RATIO_X`             | 0.40    | Fraction of cells that start as X agents             |
| `RATIO_O`             | 0.40    | Fraction of cells that start as O agents             |
| `SIMILARITY_THRESHOLD`| 0.30    | Minimum fraction of same-type neighbours for happiness |
| `MAX_ROUNDS`          | 30      | Hard stop even if agents are still unhappy           |

Cells not assigned to X or O start empty (`RATIO_X + RATIO_O` must be ≤ 1).

### Things worth trying

- Raise the threshold to 0.5 or 0.7 — watch how quickly segregation appears.
- Make the two groups unequal in size.
- Increase empty space (lower both ratios) — does it take longer to segregate?

## Reading the grid

```
X  — happy X agent
O  — happy O agent
.  — empty cell
X°  — unhappy X agent (will move next round)
O°  — unhappy O agent (will move next round)
```

## Reading the stats

- **Unhappy agents** — how many agents want to move this round and what
  percentage of the total they represent.
- **Segregation index** — the average fraction of same-type occupied
  neighbours across all agents. Starts near 0.5 on a random grid; climbs
  toward 1.0 as clusters form.
