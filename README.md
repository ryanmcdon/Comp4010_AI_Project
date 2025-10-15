# Comp4010_AI_Project
Repository for our group project

October 15th 2025 - Progress Update

1) Define an Apotris variant
We are doing the Variant Dig on Apotris Tetris. This is a variant that consists of having lines of “garbage” already on the board, and the goal is to clear these lines of garbage to reach the bottom of the board.
Each row of garbage has exactly 1 open line that must be filled to clear the line.

2) Define a MDP 
Our MDP is precise:
	State Space:
Board configuration is Massive! If we were to track if each tile was filled with a block or not, the total number of unique state combinations would equal 2200. Therefore, we have to summarize the state space somehow to minimize the number of states.
Ideas to minimize state space:
Contours
In a university paper written by  https://www.tau.ac.il/~mansour/rl-course/student_proj/livnat/tetris.html
Top 4 rows
Another topic mentioned in the paper 
	Incoming Pieces
The state will also include the different pieces that will be given in the future. Our initial plan is to simply this to the Current Piece, Next Piece, and the Piece currently in storage.		

	Action Space:
		Down, Move Right, Move Left, Spin Clockwise
	Rewards:
		Our rewards are designed specifically for the Dig gamemode:
			Reward for Clearing a row (+10  * number of rows)
			Sitting idle, IE not clearing, not placing (-1)
			Placing a block that does not get deleted (-1 * height)

            
3) RL algos to implement
	

4) plan systematic experiments comparing your approach to existing Tetris RL work






Game states:
First assumption, for simplicity, is we are going to compute the board as if it is a 10 * 4 section from the original 10 * 20 board. We will use a formula that converts the distance from the level of a column to the left to the column to the right to make a number. For example, the first column is always 
