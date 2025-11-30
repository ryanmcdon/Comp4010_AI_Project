# TetrisGreedy.py
#
# Greedy one-step policy for your Unity Tetris environment.
# - Parses obs from ML-Agents
# - Reconstructs board and active piece
# - Simulates each action 0..5
# - Uses a Tetris heuristic to pick the best action

import numpy as np

H = 20  # board height
W = 10  # board width

# -------------------------------------------------
# OBS PARSING
# -------------------------------------------------

def parse_obs(obs):
    """
    obs layout (from your TetrisAgent):
      - first 200: flattened grid (values / 3.0)
      - next 1: pieceId normalized as (id+1)/8.0
      - next 2: normalLinesCleared/100, garbageLinesCleared/100

    Returns:
      board_static: HxW np.array, ints 0/1 (occupied)
      piece_cells:  list of (r, c) for active piece (value==3)
    """
    # 1) grid values
    grid_vals = obs[: H * W] * 3.0
    grid_int = np.rint(grid_vals).astype(int).reshape(H, W)

    piece_cells = []
    board_static = np.zeros((H, W), dtype=int)

    for r in range(H):
        for c in range(W):
            v = grid_int[r, c]
            if v == 3:
                # active piece
                piece_cells.append((r, c))
            elif v != 0:
                # any non-zero (1=locked, 2=garbage) = occupied
                board_static[r, c] = 1

    return board_static, piece_cells


# -------------------------------------------------
# BASIC GEOMETRY HELPERS
# -------------------------------------------------

def can_move(piece_cells, dr, dc, board):
    """Check if piece can move by (dr, dc) without collision or out of bounds."""
    for (r, c) in piece_cells:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            return False
        if board[nr, nc] != 0:
            return False
    return True


def move_piece(piece_cells, dr, dc):
    """Return a new list of cells moved by (dr, dc)."""
    return [(r + dr, c + dc) for (r, c) in piece_cells]


def rotate_piece(piece_cells, board):
    """
    Approximate 90° clockwise rotation around the piece centroid.
    If rotation is invalid (collision / OOB), return original cells.
    """
    if not piece_cells:
        return piece_cells

    # centroid
    rs = np.array([r for (r, _) in piece_cells], dtype=float)
    cs = np.array([c for (_, c) in piece_cells], dtype=float)
    rc = rs.mean()
    cc = cs.mean()

    rotated = []
    for (r, c) in piece_cells:
        dy = r - rc
        dx = c - cc
        # 90° clockwise: (dx, dy) -> (dy, -dx)
        ndy = dx
        ndx = -dy
        nr = int(round(rc + ndy))
        nc = int(round(cc + ndx))
        rotated.append((nr, nc))

    # validate
    # also check for duplicates (bad rotation)
    if len(set(rotated)) != len(rotated):
        return piece_cells

    for (r, c) in rotated:
        if r < 0 or r >= H or c < 0 or c >= W:
            return piece_cells
        if board[r, c] != 0:
            return piece_cells

    return rotated


def drop_piece(piece_cells, board):
    """Hard drop: move piece down (towards r=0) as far as possible."""
    if not piece_cells:
        return piece_cells

    current = list(piece_cells)
    # down in our indexing is r-1 (bottom row is r=0)
    while can_move(current, -1, 0, board):
        current = move_piece(current, -1, 0)
    return current


# -------------------------------------------------
# LINE CLEAR + SCORING
# -------------------------------------------------

def lock_piece(board, piece_cells):
    """Place piece cells into board as occupied (1)."""
    new_board = board.copy()
    for (r, c) in piece_cells:
        if 0 <= r < H and 0 <= c < W:
            new_board[r, c] = 1
    return new_board


def clear_full_lines(board):
    """
    Remove fully-filled rows and drop above rows down.
    Returns (new_board, lines_cleared).
    """
    lines_cleared = 0
    new_rows = []
    for r in range(H):
        if np.all(board[r, :] != 0):
            lines_cleared += 1
        else:
            new_rows.append(board[r, :].copy())

    # pad with empty rows at the top
    while len(new_rows) < H:
        new_rows.append(np.zeros(W, dtype=int))

    new_board = np.vstack(new_rows)
    return new_board, lines_cleared


def evaluate_state(board, piece_cells=None, lines_cleared=0):
    """
    Heuristic evaluation of a board state.
    Negative weights on:
      - total column heights
      - holes
      - bumpiness
    Positive weight on lines cleared (if any).
    """
    mat = board.copy()

    # optionally include active piece as filled
    if piece_cells is not None:
        for (r, c) in piece_cells:
            if 0 <= r < H and 0 <= c < W:
                mat[r, c] = 1

    # column heights
    heights = np.zeros(W, dtype=int)
    for c in range(W):
        col = mat[:, c]
        # highest occupied cell
        non_zero = np.where(col != 0)[0]
        if len(non_zero) > 0:
            heights[c] = non_zero[-1] + 1  # +1 since rows start at 0

    total_height = heights.sum()

    # holes: empty cell below highest filled in each column
    holes = 0
    for c in range(W):
        col = mat[:, c]
        non_zero = np.where(col != 0)[0]
        if len(non_zero) > 0:
            top = non_zero[-1]
            # any zeros below 'top' are holes
            holes += np.sum(col[:top] == 0)

    # bumpiness
    bumpiness = np.sum(np.abs(np.diff(heights)))

    # standard-ish Tetris AI weights
    score = (
        -0.51066 * total_height
        -0.76066 * holes
        -0.35663 * bumpiness
        + 1.0 * lines_cleared
    )

    return score


# -------------------------------------------------
# ACTION SIMULATION
# -------------------------------------------------

def simulate_action(board, piece_cells, action):
    """
    Simulate a single action:
      0 = No-op
      1 = Left
      2 = Right
      3 = Rotate
      4 = Soft drop (one step)
      5 = Hard drop (lock + clear lines)

    Returns:
      new_board, new_piece_cells, lines_cleared
    """
    if not piece_cells:
        # no active piece -> nothing meaningful to do
        return board, piece_cells, 0

    board_base = board.copy()
    pc = list(piece_cells)

    if action == 0:
        # no-op
        return board_base, pc, 0

    elif action == 1:
        # left
        if can_move(pc, 0, -1, board_base):
            pc = move_piece(pc, 0, -1)
        return board_base, pc, 0

    elif action == 2:
        # right
        if can_move(pc, 0, 1, board_base):
            pc = move_piece(pc, 0, 1)
        return board_base, pc, 0

    elif action == 3:
        # rotate
        pc = rotate_piece(pc, board_base)
        return board_base, pc, 0

    elif action == 4:
        # soft drop (one row down)
        if can_move(pc, -1, 0, board_base):
            pc = move_piece(pc, -1, 0)
        return board_base, pc, 0

    elif action == 5:
        # hard drop + lock + clear lines
        pc = drop_piece(pc, board_base)
        locked_board = lock_piece(board_base, pc)
        cleared_board, lines_cleared = clear_full_lines(locked_board)
        # after hard drop, piece is effectively gone (new piece will spawn)
        return cleared_board, [], lines_cleared

    # fallback
    return board_base, pc, 0


# -------------------------------------------------
# GREEDY POLICY
# -------------------------------------------------

def greedy_policy(obs):
    """
    Main entrypoint: given obs from Unity, pick best action 0..5.
    """
    board, piece_cells = parse_obs(obs)

    # if somehow no active piece, just hard-drop garbage / do nothing
    if not piece_cells:
        return 5  # or 0

    best_action = 0
    best_score = -1e9

    for a in range(6):
        new_board, new_piece, lines_cleared = simulate_action(board, piece_cells, a)
        # when not hard-dropping, still evaluate with active piece present
        if a == 5:
            score = evaluate_state(new_board, None, lines_cleared)
        else:
            score = evaluate_state(new_board, new_piece, lines_cleared)

        if score > best_score:
            best_score = score
            best_action = a

    return best_action
