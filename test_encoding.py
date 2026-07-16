import numpy as np
import chess
from encoding import _fill_piece_planes


#testing whether white pawns and black pawns are in the right spaces
def test_fill_pieces_planes():
    board = chess.Board()
    a = np.zeros((102, 8, 8), dtype=np.float32)
    b = 0
    _fill_piece_planes(board, a, b)
    assert np.array_equal(
        a[0, 1], np.ones(8)) and np.array_equal(a[6,6], np.ones(8)
        )
