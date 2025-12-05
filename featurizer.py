import numpy as np

# Pre-computed mask for 4x4 area with top-middle 2x2 removed (for performance)
# This mask removes indices [0,1], [0,2], [1,1], [1,2] to keep 12 outer squares
_MASK_4X4_NO_CENTER = np.ones((4, 4), dtype=bool)
_MASK_4X4_NO_CENTER[0:2, 1:3] = False  # Remove top-middle 2x2

# Pre-computed power arrays for vectorized base conversion (for performance)
_POWERS_BASE_9 = 9 ** np.arange(9, dtype=np.int64)  # For base-9 contour encoding
_POWERS_BASE_2_25 = np.uint64(2) ** np.arange(25, dtype=np.uint64)  # For 5x5 binary encoding
_POWERS_BASE_2_16 = 1 << np.arange(15, -1, -1, dtype=np.int64)  # For 4x4 binary encoding (MSB first)
_POWERS_BASE_2_12 = 1 << np.arange(11, -1, -1, dtype=np.int64)  # For 12-bit encoding (MSB first)

#This file features different featurizer methods along with functions to get the data that will be featurized
# We will use these to reduce state space

#1) Featurize with contour + related functions
def board_contour_unbounded(board):
    """
    Given a 20x10 numpy array, return the 'contour' as height differences between columns.
    Returns unbounded differences (not clipped to [-3, +2]).
    Returns a length-9 array of differences between adjacent columns.
    Uses the bottommost filled cell (max row index) for each column, like the original.
    Both 0's and 3's are considered blank spaces - only values > 0 and != 3 are considered filled.
    
    Vectorized implementation for better performance.
    """
    board = np.asarray(board)
    rows, cols = board.shape
    
    # Create mask for filled cells (not 0 and not 3)
    filled_mask = (board > 0) & (board != 3)
    
    # Get column heights: for each column, find the max row index with filled cell
    # If no filled cells, use `rows` as the height
    col_indices = np.arange(rows).reshape(-1, 1)  # Shape (rows, 1)
    
    # For each column, get max row index where filled, or `rows` if empty
    heights = np.where(
        filled_mask.any(axis=0),
        np.where(filled_mask, col_indices, -1).max(axis=0),
        rows
    )
    
    # Calculate differences between adjacent columns
    contour = heights[:-1] - heights[1:]
    return contour


def _contour_to_state_number(contour):
    """
    Helper function: Convert a contour array to a state number without canonicalization.
    Uses base-9 encoding with clipping to [-4, +4].
    
    Vectorized implementation using pre-computed powers for better performance.
    """
    max_len = 9

    # Truncate or pad contour to length 9
    contour = np.asarray(contour)
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

    # Vectorized conversion using pre-computed powers (dot product instead of loop)
    state_number = np.dot(mapped, _POWERS_BASE_9)

    return int(state_number)


def featurize_board(board, piece_id=None, move_number=None, n_bins=1000):
    """
    Featurize a 20x10 Tetris board into a state index in range [0, n_bins-1].
    Uses 3 features: contour state ID, active piece ID, and move number.
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        piece_id: Current active piece ID (optional, defaults to 0 if None)
        move_number: Move number (0-5) (optional, defaults to 0 if None)
        n_bins: Number of bins for state space discretization
    
    Returns:
        tuple: (state_index, has_piece) where:
            state_index: Integer state index in range [0, n_bins-1]
            has_piece: Boolean indicating if a piece (value 3) was found on the board
    """
    board = np.asarray(board).reshape((20, 10))
    
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
    
    # Check if piece (value 3) exists on the board
    has_piece = np.any(board == 3)
    
    return state_index, has_piece


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
    board = np.asarray(board).reshape((20, 10))
    rows, cols = board.shape
    
    # Find all positions where the piece (value 3) is located
    piece_positions = np.where(board == 3)
    
    if len(piece_positions[0]) == 0:
        # No piece found, return zeros
        print("No piece found, returning zeros")
        print_array(board)
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
    print_array(area_5x5)
    return area_5x5


def _column_heights_and_holes(board):
    """
    Compute column heights and hole counts for a 20x10 board.
    Treats any non-zero cell (including active piece marked as 3) as filled.
    Returns:
        heights (np.ndarray shape (10,)): height of each column measured from bottom (0-20)
        holes (int): total number of empty cells that have at least one filled cell above them
    """
    board = np.asarray(board).reshape((20, 10))
    rows, cols = board.shape
    heights = np.zeros(cols, dtype=int)
    holes = 0
    for c in range(cols):
        col = board[:, c]
        filled_indices = np.where(col != 0)[0]
        if filled_indices.size == 0:
            heights[c] = 0
            continue
        top_index = filled_indices[0]  # smallest row index with a filled cell (topmost filled)
        heights[c] = rows - top_index
        # Holes are empty cells below the topmost filled cell
        below = col[top_index:]
        holes += int(np.sum(below == 0))
    return heights, holes


def featurize_tabular_state(board, piece_id=0, locked_flag=0, extra_value=0):
    """
    Compact tabular featurizer for a 20x10 Tetris board.
    Returns a discrete state key (tuple) suitable for dictionary-based Q-tables.

    Args:
        board: 20x10 array where 1/2 are placed blocks and 3 is active piece.
        piece_id: integer identifier of the current tetromino.
        locked_flag: integer flag indicating if a piece was locked on the last step.
        extra_value: an additional integer from the environment (e.g., lines cleared).

    Features (all discretized coarsely to keep table size small):
        - max_height_bin: bucketed max column height
        - total_height_bin: bucketed sum of column heights
        - holes_bin: bucketed total holes
        - bumpiness_bin: bucketed sum of |height_i - height_{i+1}|
        - piece_id (int)
        - locked_flag (0/1)
        - extra_value (clamped small int)
    """
    board = np.asarray(board).reshape((20, 10))
    heights, holes = _column_heights_and_holes(board)
    max_height = int(np.max(heights))
    total_height = int(np.sum(heights))
    bumpiness = int(np.sum(np.abs(np.diff(heights))))

    # Bin edges tuned to keep the table compact while retaining structure
    max_height_bin = int(np.digitize(max_height, [4, 8, 12, 16, 20]))
    total_height_bin = int(np.digitize(total_height, [10, 20, 40, 60, 80, 120]))
    holes_bin = int(np.digitize(holes, [0, 2, 5, 10, 20, 40]))
    bumpiness_bin = int(np.digitize(bumpiness, [0, 3, 6, 10, 15, 25]))

    locked_flag = int(bool(locked_flag))
    piece_id = int(piece_id) if piece_id is not None else 0
    extra_value = int(np.clip(extra_value if extra_value is not None else 0, -2, 10))

    # Tuple key keeps Q-table dictionary small and hashable
    return (
        max_height_bin,
        total_height_bin,
        holes_bin,
        bumpiness_bin,
        piece_id,
        locked_flag,
        extra_value,
    )

def _5x5_to_state_number(area_5x5):
    """
    Helper function: Convert a 5x5 area to a state number.
    Treats each cell as binary (true/false): True (1) if cell is filled (value > 0),
    False (0) if cell is empty (0).
    The piece (value 3) is included in the extraction, so position/rotation information is preserved.
    Flattens the 5x5 array (25 values) and encodes as binary number (base-2).
    
    Vectorized implementation using pre-computed powers for better performance.
    """
    area_5x5 = np.asarray(area_5x5).flatten()  # Flatten to 25 values
    max_len = 25
    
    # Ensure we have exactly 25 values
    if area_5x5.shape[0] > max_len:
        area_flat = area_5x5[:max_len]
    elif area_5x5.shape[0] < max_len:
        area_flat = np.pad(area_5x5, (0, max_len - area_5x5.shape[0]), 'constant')
    else:
        area_flat = area_5x5
    
    # Convert to binary: True (1) if cell is filled (value > 0), False (0) if empty
    binary = (area_flat > 0).astype(np.uint64)
    
    # Vectorized conversion using pre-computed powers (dot product instead of loop)
    state_number = np.dot(binary, _POWERS_BASE_2_25)
    
    return int(state_number)


def featurize_board_5x5(board, piece_id=0, move_number=0, n_bins=100):
    print("Featurizing board with 5x5 area around piece")
    """
    Featurize a 20x10 Tetris board by extracting a 5x5 area around the piece.
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        piece_id: Current active piece ID (default: 0)
        move_number: Move number (default: 0)
        n_bins: Number of bins for state space discretization (default: 100)
        
    Returns:
        tuple: (state_index, has_piece) where:
            state_index: Integer state index in range [0, n_bins-1]
            has_piece: Boolean indicating if a piece (value 3) was found on the board
    """
    pieces = 7
    board = np.asarray(board).reshape((20, 10))
    
    # Extract 5x5 area around piece (including the piece itself for position/rotation learning)
    area_5x5 = extract_5x5_around_piece(board)
    
    # Convert 5x5 area to state number using binary encoding
    area_state = _5x5_to_state_number(area_5x5)  # Range: 0 to 2^25-1
    
    if (n_bins % pieces != 0):
        print("n_bins must be divisible by pieces")
        return None, False
    else:      
        # Hash to n_bins using modulo with prime multiplier for better distribution
        state_index = (area_state * 31) % (n_bins // pieces)
        state_index = int(state_index * pieces + piece_id)
        
        # Check if piece (value 3) exists on the board
        has_piece = np.any(board == 3)
        
        return state_index, has_piece
    
#3 No featurizer... take 4x4

#extract 4x4 around piece
def extract_4x4_around_piece(board):
    """
    Given a 20x10 numpy array (Tetris board), find and return the 4x4 grid
    containing the active piece (cells with value 3).
    If multiple '3's are present, returns the region around the first found.
    If no '3's, returns a 4x4 grid at (0,0).
    """
    board = np.flipud(np.asarray(board).reshape((20, 10)))
    piece_indices = np.argwhere(board == 3)
    if piece_indices.shape[0] == 0:
        # No active piece found; return top-left 4x4 as fallback
        return board[:4, :4].copy()
    # Get min & max rows/cols containing any part of piece
    row_min = np.min(piece_indices[:,0])
    row_max = np.max(piece_indices[:,0])
    col_min = np.min(piece_indices[:,1])
    col_max = np.max(piece_indices[:,1])
    # Center the 4x4 window around the mean row/col of the piece, but fit into board
    row_c = int(np.mean([row_min, row_max]))
    col_c = int(np.mean([col_min, col_max]))
    # Get window top-left so that piece is centered if possible
    row_start = max(0, min(board.shape[0]-4, row_c-1))
    col_start = max(0, min(board.shape[1]-4, col_c-1))
    #print_array(board[row_start:row_start+4, col_start:col_start+4].copy())
    return board[row_start:row_start+4, col_start:col_start+4].copy()


def print_array(arr):
    """
    Utility test function to print a numpy array (typically a board) to the console
    with a readable format.
    """
    arr = np.asarray(arr)
    for row in arr:
        print(' '.join(str(x) for x in row), flush=True)
    print()  # Add blank line after array for readability


#find orientation of piece using the piece_id
def find_orientation(board, piece_id):
    """
    Find the orientation of a piece.
    """
    return piece_id

#find position of piece
def find_position(board, piece_id):
    """
    Find the position of a piece.
    """
    return piece_id


def featurize_board_4x4(board, piece_id=0, move_number=0, n_bins=100):
    """
    Featurize a 20x10 Tetris board by extracting a 4x4 area around the piece.
    Returns a tuple (state_index, has_piece) where:
        state_index: Integer state index in range [0, n_bins-1]
        has_piece: Boolean indicating if a piece (value 3) was found on the board
    
    Vectorized binary conversion for better performance.
    """
    board_array = np.asarray(board).reshape((20, 10))
    # Check if piece (value 3) exists on the board before extraction
    has_piece = np.any(board_array == 3)
    
    area_4x4 = extract_4x4_around_piece(board)
    # Convert the 4x4 area to a unique number using only zero/nonzero (binary)
    area_binary = (area_4x4 > 0).astype(np.int64)
    flat_binary = area_binary.flatten()
    
    # Vectorized binary to int conversion using pre-computed powers (MSB first)
    unique_number = np.dot(flat_binary, _POWERS_BASE_2_16)
    
    # Flip the 4x4 area horizontally (left-right)
    flipped_area_4x4 = np.fliplr(area_binary)
    flat_flipped = flipped_area_4x4.flatten()
    
    # Vectorized binary to int for flipped
    flipped_number = np.dot(flat_flipped, _POWERS_BASE_2_16)
    
    # Take the lower of the two numbers (original, flipped)
    if unique_number <= flipped_number:
        area_4x4_val = unique_number
        has_piece = False
    else:
        area_4x4_val = flipped_number
        has_piece = True
    
    if area_4x4_val > n_bins:
        area_4x4_val = area_4x4_val % n_bins
    
    return int(area_4x4_val), has_piece


# Piece ID mapping for state space reduction (pieces 4 and 6 share states with 3 and 5)
PIECE_ID_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4}  # Maps piece_id to shift value
PIECE_FLIP_MAP = {4, 6}  # Pieces that flip has_piece flag

def featurize_board_4x4_no_center(board, piece_id=0, move_number=0, n_bins=100):
    """
    Featurize a 20x10 Tetris board by extracting a 4x4 area around the piece,
    then removing the center 2x2 squares, creating a 12-bit state from the remaining 12 squares.
    
    Returns a tuple (state_index, has_piece) where:
        state_index: Integer state index in range [0, n_bins-1]
        has_piece: Boolean indicating if a piece (value 3) was found on the board
    
    Args:
        board: 20x10 numpy array representing the Tetris board
        piece_id: Current active piece ID (default: 0)
        move_number: Move number (default: 0)
        n_bins: Number of bins for state space discretization (default: 100)
    
    Vectorized binary conversion for better performance.
    """
    board_array = np.asarray(board).reshape((20, 10))
    # Check if piece (value 3) exists on the board before extraction
    has_piece = np.any(board_array == 3)
    
    area_4x4 = extract_4x4_around_piece(board)
    
    # Remove top-middle 2x2 squares (indices [0,1], [0,2], [1,1], [1,2])
    # This keeps more information about the bottom rows
    # Use pre-computed mask for performance (created once at module load)
    mask = _MASK_4X4_NO_CENTER
    
    # Extract the 12 squares (outer ring)
    outer_squares = area_4x4[mask]  # This gives us 12 values
    
    # Convert to binary: True (1) if cell is filled (value > 0), False (0) if empty
    area_binary = (outer_squares > 0).astype(np.int64)
    
    # Vectorized binary to int conversion using pre-computed powers (MSB first)
    unique_number = np.dot(area_binary, _POWERS_BASE_2_12)
    
    # Flip horizontally and compare (for symmetry)
    # Reconstruct the 4x4 with center removed to flip it
    area_4x4_with_mask = np.zeros((4, 4), dtype=np.int64)
    area_4x4_with_mask[mask] = outer_squares
    flipped_area_4x4 = np.fliplr(area_4x4_with_mask)
    flipped_outer_squares = flipped_area_4x4[mask]
    flipped_binary = (flipped_outer_squares > 0).astype(np.int64)
    
    # Vectorized binary to int for flipped
    flipped_number = np.dot(flipped_binary, _POWERS_BASE_2_12)
    
    
    # Take the lower of the two numbers (original, flipped) for symmetry
    # This helps reduce state space but doesnt reduce range that can be returned its still  0 to 2*12-1
    # Instead some numbers will now never be returned. There is no simple way to know which numbers 
    # meaning no space saving but there is time saving!
    state_index = min(unique_number, flipped_number)
    
    # Update has_piece based on which one was used (if flipped was used, board was flipped)
    if flipped_number < unique_number:
        has_piece = True  # Indicates flipped version was used
    
    #Some pieces are equal to each other if the board was fliped (again)
    #Here I use this fact to reduce state space futher
    
    match piece_id:
        case 0:
            state_index = (0 << 12) | state_index; #I piece
        case 1:
            state_index = (1 << 12) | state_index; #O piece
        case 2:
            state_index = (2 << 12) | state_index; #T piece
        case 3:
            state_index = (3 << 12) | state_index; #S piece
        case 4:
            state_index = (3 << 12) | state_index; #Z piece
            has_piece = not has_piece;
        case 5: 
            state_index = (4 << 12) | state_index; #L piece
        case 6:
            state_index = (4 << 12) | state_index; #J piece
            has_piece = not has_piece;
    # Map to n_bins 
    
    state_index = state_index % n_bins
    
    #Total state space should be    = (Piece state_space * board state_space)
    #                               = 2^3 - (3) * 2^12 / 2  <= remove un-used -piece number + roughly half of numbers now invalid returns
    #                               = 5 * 2048
    #                               = 10,240 but still need about double that for storage
    
    return state_index, has_piece
    
    
    #ideas:
    # somehow include the piece orientation and remove the piece first so we can reduce state space
    # Instead of center blocks remove top row so that changes the area the AI can see? and still remove top 2 blocks of square 
    # A view like this?
    #           x   x   x   x
    #           1   x   x   1
    #           1   1   1   1
    #           1   1   1   1
