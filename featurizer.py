import numpy as np


def board_contour_unbounded(board):
    """
    Given a 20x10 numpy array, return the 'contour' as height differences between columns.
    Returns unbounded differences (not clipped to [-3, +2]).
    Returns a length-9 array of differences between adjacent columns.
    Uses the bottommost filled cell (max row index) for each column, like the original.
    Both 0's and 3's are considered blank spaces - only values > 0 and != 3 are considered filled.
    """
    board = np.array(board)
    rows, cols = board.shape
    contour = np.zeros(cols - 1, dtype=int)
    prev_height = None
    
    for col in range(cols):
        column = board[:, col]
        # Consider both 0's and 3's as blank spaces - only count filled cells that are > 0 and != 3
        filled = np.where((column > 0) & (column != 3))[0]
        # print("Column:", column)
        # print("Filled:", filled)
        # Use max (bottommost) filled cell, or rows if column is empty
        current_height = filled.max() if len(filled) > 0 else rows
        
        if col > 0:
            if prev_height is not None:
                # Calculate difference: positive means current column is lower (higher row index)
                # This matches the original logic: prev - current
                diff = prev_height - current_height
                contour[col - 1] = diff
        prev_height = current_height
    
    return contour


def _contour_to_state_number(contour):
    """
    Helper function: Convert a contour array to a state number without canonicalization.
    Uses base-9 encoding with clipping to [-4, +4].
    """
    base = 9
    max_len = 9

    # Truncate or pad contour to length 9
    contour = np.array(contour)
    if contour.shape[0] > max_len:
        contour_short = contour[:max_len]
    elif contour.shape[0] < max_len:
        # pad with zeros
        contour_short = np.pad(contour, (0, max_len - contour.shape[0]), 'constant')
    else:
        contour_short = contour

    # Map from [-4, +4] -> [0, 8] (base-9 encoding)
    # Clip unbounded values to the range [-4, +4]
    mapped = np.clip(contour_short, -4, 4) + 4  # range now [0, 8]

    # Convert to state number (base-9 number)
    state_number = 0
    for i, val in enumerate(mapped):
        state_number += int(val) * (base ** i)

    return state_number


def featurize_board(board, piece_id=None, n_bins=1000):
    """
    Featurize a 20x10 Tetris board into a state index in range [0, n_bins-1].
    Uses only 2 features: contour state ID and active piece ID.
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        piece_id: Current active piece ID (optional, defaults to 0 if None)
        n_bins: Number of bins for state space discretization
    """
    board = np.array(board).reshape((20, 10))
    
    # Normalize piece_id (handle None case and ensure it's an integer)
    if piece_id is None:
        piece_id = 0
    else:
        piece_id = int(piece_id)

    # Get contour and compute state ID using old contour state function
    contour = board_contour_unbounded(board)  # Length 9 array
    contour_state = _contour_to_state_number(contour)

    # Only 2 features: contour_state and piece_id
    feature_vec = np.array([
        contour_state,                  # Contour state number (base-9 encoding, range [-4, +4])
        piece_id,                       # Active piece ID
    ])

    # Normalize features
    # Contour state: base-9 encoding with 9 values = 9^9 possible states
    # Piece ID: assuming max 10 piece types
    max_values = np.array([
        9**9,     # contour_state (base-9 encoding, 9 values: 0 to 9^9-1)
        10,       # piece_id (normalize assuming max 10 piece types)
    ]) + 1e-8
    
    norm_vec = np.clip(feature_vec / max_values, 0, 1)

    # Hashing: quantize to bins per feature for higher resolution
    quantized = np.clip((norm_vec * 10).astype(int), 0, 9)
    # Create hash code using base-10 encoding
    hash_code = 0
    base = 1
    for val in quantized:
        hash_code += base * val
        base *= 10
    
    # Use modulo with prime multiplier for better distribution
    state_index = (hash_code * 31) % n_bins
    return state_index

