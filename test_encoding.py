import numpy as np
import chess
from encoding import _fill_piece_planes, _perspective, board_to_tensor


#testing whether white pawns and black pawns are in the right spaces
def test_fill_pieces_planes():
    board = chess.Board()
    a = np.zeros((102, 8, 8), dtype=np.float32)
    b = 0
    _fill_piece_planes(board, a, b)
    assert np.array_equal(
        a[0, 1], np.ones(8)) and np.array_equal(a[6,6], np.ones(8)
        )

#testing whether e4 pawn is in the correct location when board is flipped
def test_perspective():
    board = chess.Board()
    board.push_san("e4")
    board = _perspective(board, True)
    a = np.zeros((102, 8, 8), dtype=np.float32)
    _fill_piece_planes(board, a, 0)
    assert a[6][4, 4] == 1.0



def test_board_to_tensor_start():
    t = board_to_tensor(chess.Board())
    #make sure shape is (102, 8 ,8)
    assert t.shape == (102, 8, 8)
    # white pawns on plane 0 rank 2 (row 1); white king plane 5 at e1 = (0, 4)
    assert np.array_equal(t[0, 1], np.ones(8))
    assert t[5, 0, 4] == 1.0
    # black (opponent) king plane 11 at e8 = (7, 4)
    assert t[11, 7, 4] == 1.0
    # side-to-move plane all ones (white to move) and all castling rights set
    assert np.array_equal(t[12], np.ones((8, 8)))
    for p in (13, 14, 15, 16):
        assert np.array_equal(t[p], np.ones((8, 8)))
    # no en passant square, and no history at the start of the game
    assert t[17].sum() == 0
    assert t[18:].sum() == 0





