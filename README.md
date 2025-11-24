# Comp4010_AI_Project
Repository for our COMP 4010 final group project

### Contributors:
- Ryan McDonald
- Stanny Huang
- Aaron Maagdenberg

TA: Paul

# October 15th 2025 - Progress Update

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
	
In terms of real-time algorithms we can look into implementing for this project, we were considering Monte Carlo and Temporal Difference as options, but we will likely try more algorithms to see which ones work the best. These algorithms have policy values included, which is extremely important for our goal as we want to prioritize our AI to be as efficient in its block placements as possible. Therefore, the AI should be rewarded more for clearing lines of garbage earlier rather than later.

## 4) Plan systematic experiments

These past two weeks, we have pivoted from editing the Apostris code directly, and instead shifting to creating an external Python application to scan the screen and make decisions based off what a player would be able to see (the state of the board, the incoming blocks, etc).

So far, we have created a script that scans the screen, a script that displays the info of what the program is seeing, and finally a script for creating an overlay so in the future we can show exactly what the AI is scanning for/making decisions off of.

In the next 2 weeks, we plan on finalizing the screen-scanning code to finish our environment and properly detect the gamestate of the Tetris board.

# October 30th 2025 - Progress Update

**Ryan**
- Implemented an improved ApostrisAnalyzer that scans the screen and parses a 1D array to store locations of blocks within the grid
- Implemented an algorithm to detect contours along the surface of the Tetris blocks
- Created the StateSpace consisting of the block locations and contouring data
- Contributed information to the Environment Demo powerpoint and video

**Aaron**
- Implemented python script that can execute keyboard inputs (allowing for Action space)
- Refactored ApostrisAnalyzer code to be cleaner + easier to read
- Contributed information to the Environment Demo powerpoint and video
- Created architecture for the implementation of the env demo states, rewards 

**Stanny**
- Edited and compiled the Environment Demo video
- Completed full screen visualization overlay in "ScreenOverlay.py"
- Contributed information to the Environment Demo powerpoint and video

To-Dos:
- We have issues regarding the Apostris anticheat when sending inputs into the game (limited by frames in-between inputs, or else flagged for cheating)
- Scanning pixel values for contouring is extremely janky (requires scan for exact pixel value of block outlines)
- Overall runtime is very slow due to constant screen scanning
We need to find a long-term solution to quell these environment issues and have the ability to work on actual AI implementation

# November 15th 2025 - Progress Update

The past 2 weeks have been extremely daunting for us, as we ran into an issue with using a screen scanner for Apostris as our environment. We had issues where some blocks wouldn't be picked up from scans, as well as the extremely-poor runtime of constantly parsing data from pixel-perfect screen scanning. The most glaring issue however was the fact that there was a built-in anti-cheat of Apotris that prevents players from making macro inputs every frame, meaning we could not have our AI interact with the game at peak efficiency. Therefore, we ultimately the hard decision to completely pivot off of Apostris for our environment, and swap entirely to a new system revolving around our own Tetris game, built from the ground up using Unity 2020.3.49f1. <br> <br>
This makes the previous python file of ApostrisAnalyzer deprecated. However, our AI is still being programmed in Python; our new method is to create a Socket-Signal server with the host being the new "StateServer.py" python file and the client being the Unity Tetris recreation. The Tetris client connects to the python file and sends JSON data of the state-space for our AI to use.
Link to house-made Tetris repo: https://github.com/littlemanstann/COMP4010-UnityTetrisRecreation 

Stanny
- Created a recreation of Tetris using Unity, replicating the Apostris Dig gamemode
- Completed the refactor of our system to transfer data from the Tetris client and python project
- Created a Socket-Signal server setup between python file "StateServer.py" and Unity Tetris executable

Aaron
- Developed new functionality logic for board size detection, to be universal for all resolutions, window sizes etc
- Optimization and organization to pre-exsisting code base
- Error fixing misc screen analyzer bugs

Ryan
- Bug fixes to some edge cases in ApotrisAnalyser
- Setup more reward functions for future improvments and use

To-Dos:
- Start AI devlopment!!
- Implement MDP AI with our new environment
- Have Unity Tetris game send contour data to python files as a state-space
- Implement proper action space based off contour state-space
- Test various reward options:
	- Height of tallest block (smaller height = greater reward)
 	- Lines cleared
  	- Time taken per line clear
