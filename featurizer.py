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


def featurize_board(board, piece_id=None, move_number=None, n_bins=1000):
    """
    Featurize a 20x10 Tetris board into a state index in range [0, n_bins-1].
    Uses 3 features: contour state ID, active piece ID, and move number.
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        piece_id: Current active piece ID (optional, defaults to 0 if None)
        move_number: Move number (0-5) (optional, defaults to 0 if None)
        n_bins: Number of bins for state space discretization
    """
    board = np.array(board).reshape((20, 10))
    
    # Normalize piece_id (handle None case and ensure it's an integer)
    if piece_id is None:
        piece_id = 0
    else:
        piece_id = int(piece_id)
    
    # Normalize move_number (handle None case and ensure it's an integer in range 0-5)
    if move_number is None:
        move_number = 0
    else:
        move_number = int(move_number)
        # Ensure move_number is in valid range [0, 5]
        move_number = np.clip(move_number, 0, 8)

    # Get contour and compute state ID using old contour state function
    contour = board_contour_unbounded(board)  # Length 9 array
    contour_state = _contour_to_state_number(contour)

    # 3 features: contour_state, piece_id, and move_number
    feature_vec = np.array([
        contour_state,                  # Contour state number (base-9 encoding, range [-4, +4])
        piece_id,                       # Active piece ID
        move_number,                    # Move number (0-5)
    ])

    # Normalize features
    # Contour state: base-9 encoding with 9 values = 9^9 possible states
    # Piece ID: assuming max 10 piece types
    # Move number: range 0-5 (6 possible values)
    max_values = np.array([
        9**9,     # contour_state (base-9 encoding, 9 values: 0 to 9^9-1)
        7,       # piece_id (normalize assuming max 10 piece types)
        9,        # move_number (range 0-5, so max is 5, normalize with 6)
    ]) + 1e-8
    
    norm_vec = np.clip(feature_vec / max_values, 0, 1)

    # Hashing: quantize to bins per feature with different resolutions
    # Give more importance to piece_id and move_number by using more bins
    contour_bins = 10      # Fewer bins for contour (reduced importance)
    piece_id_bins = 20    # More bins for piece_id (increased importance)
    move_number_bins = 20 # More bins for move_number (increased importance)
    
    quantized = np.array([
        np.clip((norm_vec[0] * contour_bins).astype(int), 0, contour_bins - 1),
        np.clip((norm_vec[1] * piece_id_bins).astype(int), 0, piece_id_bins - 1),
        np.clip((norm_vec[2] * move_number_bins).astype(int), 0, move_number_bins - 1),
    ])
    
    # Create hash code using mixed-base encoding to preserve importance
    # Base sizes reflect the number of bins for each feature
    hash_code = quantized[0] + (quantized[1] * contour_bins) + (quantized[2] * contour_bins * piece_id_bins)
    
    # Use modulo with prime multiplier for better distribution
    state_index = (hash_code * 31) % n_bins
    return state_index


def extract_5x5_around_piece(board):
    """
    Extract a 5x5 area around the piece (marked as 3 on the board).
    Finds the center of all cells with value 3 and extracts a 5x5 window.
    Handles edge cases by padding with 0 if the window goes out of bounds.
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        
    Returns:
        5x5 numpy array centered around the piece, or zeros if no piece found
    """
    board = np.array(board).reshape((20, 10))
    rows, cols = board.shape
    
    # Find all positions where the piece (value 3) is located
    piece_positions = np.where(board == 3)
    
    if len(piece_positions[0]) == 0:
        # No piece found, return zeros
        return np.zeros((5, 5), dtype=int)
    
    # Calculate center of the piece
    center_row = int(np.mean(piece_positions[0]))
    center_col = int(np.mean(piece_positions[1]))
    
    # Extract 5x5 area centered around the piece
    # Calculate bounds (2 cells on each side of center)
    row_start = center_row - 2
    row_end = center_row + 3
    col_start = center_col - 2
    col_end = center_col + 3
    
    # Create a 5x5 array
    area_5x5 = np.zeros((5, 5), dtype=int)
    
    # Copy values from board, handling out-of-bounds by padding with 0
    # Keep the piece (value 3) so the AI can learn about position/rotation
    for i in range(5):
        for j in range(5):
            board_row = row_start + i
            board_col = col_start + j
            
            if 0 <= board_row < rows and 0 <= board_col < cols:
                board_value = board[board_row, board_col]
                # Keep the piece (value 3) to preserve position/rotation information
                area_5x5[i, j] = board_value
            else:
                area_5x5[i, j] = 0  # Pad with 0 if out of bounds
    
    return area_5x5


def _5x5_to_state_number(area_5x5):
    """
    Helper function: Convert a 5x5 area to a state number.
    Treats each cell as binary (true/false): True (1) if cell is filled (value > 0),
    False (0) if cell is empty (0).
    The piece (value 3) is included in the extraction, so position/rotation information is preserved.
    Flattens the 5x5 array (25 values) and encodes as binary number (base-2).
    """
    area_5x5 = np.array(area_5x5).flatten()  # Flatten to 25 values
    base = 2
    max_len = 25
    
    # Ensure we have exactly 25 values
    if area_5x5.shape[0] > max_len:
        area_flat = area_5x5[:max_len]
    elif area_5x5.shape[0] < max_len:
        area_flat = np.pad(area_5x5, (0, max_len - area_5x5.shape[0]), 'constant')
    else:
        area_flat = area_5x5
    
    # Convert to binary: True (1) if cell is filled (value > 0), False (0) if empty
    binary = (area_flat > 0).astype(int)
    
    # Convert to state number (base-2 number)
    state_number = 0
    for i, val in enumerate(binary):
        state_number += int(val) * (base ** i)
    
    return state_number


def featurize_board_5x5(board, piece_id=0, move_number=0, n_bins=100):
    """
    Featurize a 20x10 Tetris board by extracting a 5x5 area around the piece.
    Returns a state index in range [0, n_bins-1].
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        n_bins: Number of bins for state space discretization (default: 100)
        
    Returns:
        state_index: Integer state index in range [0, n_bins-1]
    """
    pieces = 7
    board = np.array(board).reshape((20, 10))
    
    # Extract 5x5 area around piece (including the piece itself for position/rotation learning)
    area_5x5 = extract_5x5_around_piece(board)
    
    # Convert 5x5 area to state number using binary encoding
    area_state = _5x5_to_state_number(area_5x5)  # Range: 0 to 2^25-1
    
    if (n_bins % pieces != 0):
        print("n_bins must be divisible by pieces")
        return None
    else:      
        # Hash to n_bins using modulo with prime multiplier for better distribution
        state_index = (area_state * 31) % (n_bins // pieces)
        state_index = int(state_index * pieces + piece_id)
        return state_index