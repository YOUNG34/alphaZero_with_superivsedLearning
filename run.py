import chess
import numpy as np
from mcts_pure import MCTSPlayer as mcts_pure
import chess.svg
#from IPython.display import SVG

class Run(object):
    def __init__(self):
        self.player = [1,2]

    def start_play(self):
        mcts_player = mcts_pure(c_puct=5, n_playout=100)
        board = chess.Board()
        #SVG(chess.svg.board(board=board))
        while True:
            if board.turn == True:
                move = input("Please input your move:")
                if chess.Move.from_uci(move) not in board.legal_moves:
                    print("Invalid move!")
                    continue
                board.push(chess.Move.from_uci(move))
                print(board)
                print("**********************************")
            else:
                move = mcts_player.get_action(board)
                board.push(chess.Move.from_uci(str(move)))
                print(board)
                print("**********************************")
            if board.is_variant_end():
                if board.turn:
                    print("winner is : White side")
                else:
                    print("winner is : Black side")

if __name__ == '__main__':
    play = Run()
    play.start_play()
