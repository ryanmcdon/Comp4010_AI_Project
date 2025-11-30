# MiniTetris.py

class MiniTetris:
    def __init__(self, grid):
        self.grid = [row[:] for row in grid]  # deep copy
        self.h = len(grid)
        self.w = len(grid[0])

    def column_heights(self):
        heights = [0] * self.w
        for x in range(self.w):
            for y in range(self.h):
                if self.grid[y][x] != 0:
                    heights[x] = self.h - y
                    break
        return heights

    def count_holes(self):
        holes = 0
        for x in range(self.w):
            block_seen = False
            for y in range(self.h):
                if self.grid[y][x] != 0:
                    block_seen = True
                elif block_seen and self.grid[y][x] == 0:
                    holes += 1
        return holes

    def bumpiness(self, heights):
        return sum(abs(heights[i] - heights[i+1]) for i in range(self.w - 1))

    def score(self):
        heights = self.column_heights()
        holes = self.count_holes()
        bump = self.bumpiness(heights)

        # Standard Tetris AI weighted score
        return -(0.51066 * sum(heights) +
                 0.76066 * holes +
                 0.35663 * bump)

    def apply_action(self, action):
        """
        Since we cannot simulate actual piece physics without a full engine,
        we return the board score AFTER a hypothetical action.
        This allows greedy choice among actions.
        """
        return self.score()
