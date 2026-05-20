# Schelling's Segregation Model

## 1. Pseudo-code

```
SET UP the grid
    Create an N × N grid.
    Place X agents, O agents, and empty cells at random
    (with given proportions).

REPEAT until no one is unhappy, or until round limit reached:

    FIND unhappy agents
        For each agent on the grid:
            Look at its 8 surrounding cells.
            Of the occupied neighbours, how many share its type?
            If that fraction is below the threshold,
                mark the agent as unhappy.

    IF no one is unhappy:
        STOP.

    MOVE unhappy agents
        For each unhappy agent (in random order):
            Pick a random empty cell.
            Move the agent there.
            The old cell becomes empty.

    REPORT the round
        Show the grid.
        Show how many agents are still unhappy.
        Show the segregation index.
```

---

## 2. Plain English

Create a grid. Place agents of two types and some empty cells at random
across it.

Take each agent in turn. Count its occupied neighbours. Count how many
of those share its type. If the share is below the threshold, mark
the agent as unhappy.

Move each unhappy agent to a randomly chosen empty cell. Leave the
others where they are.

Record the state of the grid and the number of unhappy agents.

Repeat. Stop when no agent is unhappy, or when the round limit is
reached.
