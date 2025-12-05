import numpy as np
from featurizer import featurize_board_4x4, extract_4x4_around_piece, print_array


def test_featurize_board_4x4(test_boards=None, piece_id=0, move_number=0, n_bins=100):
    """
    Test function for featurize_board_4x4.
    
    Args:
        test_boards: List of test boards (20x10 arrays). If None, uses default test cases.
        piece_id: Piece ID to use for testing (default: 0)
        move_number: Move number to use for testing (default: 0)
        n_bins: Number of bins for testing (default: 100)
    """
    if test_boards is None:
        # Create default test boards
        test_boards = []
        
        # Test 1: Empty board (no piece)
        empty_board = np.zeros((20, 10), dtype=int)
        test_boards.append(("Empty board (no piece)", empty_board))
        
        # Test 2: Board with piece at center
        board_with_piece = np.zeros((20, 10), dtype=int)
        board_with_piece[10, 5] = 3  # Piece at center
        board_with_piece[10, 4] = 3  # Piece extends left
        board_with_piece[11, 5] = 3  # Piece extends down
        board_with_piece[9, 4] = 3  # Piece extends up
        test_boards.append(("Board with piece at center", board_with_piece))
        
        # Test 3: Board with piece at top-left
        board_top_left = np.zeros((20, 10), dtype=int)
        board_top_left[0, 0] = 3
        board_top_left[0, 1] = 3
        board_top_left[1, 0] = 3
        board_top_left[2, 0] = 3
        test_boards.append(("Board with piece at top-left", board_top_left))
        
        # Test 4: Board with piece and some filled cells
        board_with_filled = np.zeros((20, 10), dtype=int)
        board_with_filled[15:20, 0:9] = 1  # Some filled cells at bottom
        board_with_filled[12, 7] = 3  # Piece above filled area
        board_with_filled[12, 8] = 3
        board_with_filled[13, 7] = 3
        board_with_filled[14, 7] = 3
        test_boards.append(("Board with piece and filled cells", board_with_filled))
        
        # Test 5: Board with piece at bottom-right
        board_bottom_right = np.zeros((20, 10), dtype=int)
        board_bottom_right[18, 8] = 3
        board_bottom_right[18, 9] = 3
        board_bottom_right[19, 8] = 3
        board_bottom_right[19, 9] = 3
        test_boards.append(("Board with piece at bottom-right", board_bottom_right))
    
    print("=" * 70)
    print(f"Testing featurize_board_4x4 with piece_id={piece_id}, move_number={move_number}, n_bins={n_bins}")
    print("=" * 70)
    
    for i, (description, board) in enumerate(test_boards, 1):
        print(f"\n--- Test {i}: {description} ---")
        print("\nFull board :")
        print_array(board)
        
        # Extract 4x4 area
        area_4x4 = extract_4x4_around_piece(board)
        print("\nExtracted 4x4 area:")
        print_array(area_4x4)
        
        # Get the featurized output
        print("\nFeaturized output:")
        state_idx, flipped_state_idx = featurize_board_4x4(board, piece_id=piece_id, move_number=move_number, n_bins=n_bins)
        print(f"State index: {state_idx}, Flipped state index: {flipped_state_idx}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_featurize_board_4x4()

