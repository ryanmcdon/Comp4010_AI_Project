#Reshaping the flat grid made things a lot easier, so all this function does is turns the flat list into a 2D list, 20X10
def reshape_grid(flat):
    ROWS = 20
    COLS = 10
    return [flat[i:i+COLS] for i in range(0, ROWS * COLS, COLS)]


#this is just a very simple exampleof how we might compute a reward, we can have multiple of these with diffrent functionalitys for diffrent ai agents
#The important stuff from this file is not actually this function, but the helper functions below that do the actual calculations
def compute_reward(prev_grid, new_grid, lines_cleared):

    prev_grid = reshape_grid(prev_grid)
    new_grid = reshape_grid(new_grid)
    reward = 0


    reward += lines_cleared * 10


    prev_holes = count_holes(prev_grid)
    new_holes = count_holes(new_grid)
    hole_diff = new_holes - prev_holes

    if hole_diff > 0:
        reward -= hole_diff * 5  

    prev_height = board_height(prev_grid)
    new_height = board_height(new_grid)

    if new_height < prev_height:
        reward += 1  
    elif new_height > prev_height:
        reward -= 1  

    return reward


#these are just examples of what information about the grid we might want to use to compute rewards. Please note they are both very simple implementations and could be improved
def count_holes(grid):
    rows = len(grid)
    cols = len(grid[0])
    holes = 0

    for col in range(cols):
        block_seen = False
        for row in range(rows):
            if grid[row][col] != 0:  
                block_seen = True
            elif block_seen and grid[row][col] == 0:
                holes += 1

    return holes


def board_height(grid):
    rows = len(grid)
    cols = len(grid[0])

    for row in range(rows):
        if any(grid[row][col] != 0 for col in range(cols)):
            return rows - row

    return 0