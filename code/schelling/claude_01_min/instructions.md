# Task: Schelling Segregation Model in Python

Build a minimal terminal-based implementation of Schelling's segregation agent-based model for teaching purposes.

## Audience
- I know the model well; my Python is limited.
- The code is for learning and teaching, so clarity matters more than cleverness.

## Constraints
- Use base Python only. No third-party libraries unless strictly necessary.
- Set up and use a virtual environment. Never make system-level changes.
- Ask permission before running anything.
- Keep the code as short and simple as possible.
- Comment generously: explain what each block does and why, in plain English.

## Behaviour
- Run the simulation iteratively in the terminal.
- After each iteration:
  - Print the grid using ASCII: `X` and `O` for the two groups, `.` for empty cells, `X°` / `O°` for unhappy agents.
  - Print stats for the round: round number, proportion of unhappy agents, and any other simple metric you think helps a learner.
  - Wait for the user to press Enter before continuing to the next iteration.
- Stop when no agents are unhappy, or after a sensible max number of rounds (e.g. 30).

## Parameters
- Expose the basics (grid size, ratio of each group, empty fraction, similarity threshold) as clearly named variables at the top of the file so I can tweak them.

## Deliverable
- Python code 