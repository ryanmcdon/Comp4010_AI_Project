import numpy as np


def board_contour_unbounded(board):


    board = np.array(board)
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=int)
    

    for col in range(cols):
        column = board[:, col]

        filled = np.where((column > 0) & (column != 3))[0]
        
        if len(filled) > 0:

            heights[col] = filled.max() + 1
        else:
            heights[col] = 0 
    

    contour_diff = np.diff(heights)
    
    return contour_diff




def featurize_board(board, piece_id):


    contour = board_contour_unbounded(board)
    

    contour_tuple = tuple(contour.astype(int))
    

    state_key = (contour_tuple, int(piece_id))
    
    return state_key

