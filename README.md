# Comp4010_AI_Project
Repository for our group project

October 15th 2025 - Progress Update

## 1) Define an Apotris variant
We are doing the Variant Dig on Apotris Tetris. This is a variant that consists of having lines of “garbage” already on the board, and the goal is to clear these lines of garbage to reach the bottom of the board.
Each row of garbage has exactly 1 open line that must be filled to clear the line.
![Apostris Dig Image](/assets/garbage.png "Apostris Dig Image")

## 2) Define a MDP 
Our MDP is as follows:
### State Space:
The board configuration for Tetris is massive! If we were to track if each tile was filled with a block or not, the total number of unique state combinations would equal 2200. Therefore, we have to summarize the state space somehow to minimize the number of states.

<b>Ideas to minimize state space:</b> <br>
<b>Contours</b>

In a Tel Aviv University Reinforcement Learning Tetris study written by Yael Bdolah and Dror Livnat, they use a technique of merely scanning for the contours of the layers visible to the top of the Tetris board. Link to study: https://www.tau.ac.il/~mansour/rl-course/student_proj/livnat/tetris.html

<b>Top 4 rows</b>

Another strategy mentioned in the previous study is scanning specifically the Top 4 Rows of the board, starting from the point with the highest block and scanning the rows downwards.

<b>Aside: Incoming Pieces </b>

The state given to the agent will also keep track of the different pieces that are currently being used. Our initial plan is to simplify this to the Current Piece, Next Piece, and the Storage Piece.

### Action Space:

	Move Right, Move Left, Spin Clockwise, Store Piece, Fast-Drop, Instant-Drop

### Rewards:

Our rewards are designed specifically for the Dig gamemode:
- Reward for Clearing a row (+10  * number of rows)
- Sitting idle, IE not clearing, not placing (-1)
- Placing a block that does not get deleted (-1 * height)

         
## 3) RL Algorithms to implement
	

## 4) Plan systematic experiments






Game states:
First assumption, for simplicity, is we are going to compute the board as if it is a 10 * 4 section from the original 10 * 20 board. We will use a formula that converts the distance from the level of a column to the left to the column to the right to make a number. For example, the first column is always 
