# Initial Environment

1. Create a square grid of n x n [parameterize this]
	- hard edges
	- edges and corner cells have fewer neighbours

2. Create and place agents randomly
	- n^2 * 90/100 agents (10% empty cell) [parameterize this]
	- agent attribute: X or O (50% of agents X, remaining O) [parameterize this]
	- randomly place agents into grid cells
	- each cell is in one of the states: empty, X or O

# Agent decision rule: this is where agents process information and decide

3. Calculate neighbour similarity
	- Moore neighbourhood (max 8 cells: 4 immediate + 4 diagonal)
	- For each agent, divide same-type neighbours by total occupied neighbours to find the ratio of similarity 
	- Empty cells do not count as a neighbour
	- If an agent has no neighbours, they are treated as happy.

4. Is the similarity below a threshold?
	- If the similarity level is below a threshold, the agent is unhappy.
		- strictly less than <
	- Set the threshold for all agents to τ. τ is a parameter.
	- Default τ is 30/100. 

# Round statistics

5. Calculate relevant statistics:
	- The number of unhappy agents (and its proportion to all agents)
	- Average similarity:	(the ratio similarity calculated in 3 averaged over all agents).
		- Isolated agents are disregarded for average similarity

6. Plot the grid.	
	- Show unhappy agents with a bar: x̄, ō
	- empty cells shown as . 
	- Report the statistics in the bottom
	- Put a title to report the round number and τ and initial parameters. 

7. Stop or continue:
	- If there are unhappy agents: ask user to press enter to go to the next round. 
	- If there are no unhappy agents remaining, stop.

# Next round

8. Move unhappy agents to an empty cell randomly
	- Shuffle unhappy agents and current empty cells

# Loop

9. Repeat 3-8 until a maximum round of iterations reached (parameterize this to 200)

# Default Parameters:
  • n =  20
  • density (fraction of cells filled) = 0.90
  • share_X (fraction of agents type X) = 0.50
  • τ (happiness threshold) = 0.30
  • max_rounds (iteration cap) = 200