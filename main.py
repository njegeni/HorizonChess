# example of python-chess library
import chess
import chess.pgn

# board = chess.Board()
# board.push_san("e4")
# print(board)
# print(board.turn) #True for White, False for Black
# print(list(board.legal_moves))

with open("pgnmentor.pgn", mode="r", encoding="utf-8") as pgn_file:

    game = chess.pgn.read_game(pgn_file)

    if game is not None:
        # 1. Extract Meta Information (Headers)
        print("Event:", game.headers.get("Event"))
        print("White Player:", game.headers.get("White"))
        print("Black Player:", game.headers.get("Black"))
        print("Result:", game.headers.get("Result"))
        
        # 2. Iterate through and display all moves in Standard Algebraic Notation (SAN)
        print("\nMove History:")
        for move in game.mainline_moves():
            print(move)  # Outputs standard move object or can be converted to SAN