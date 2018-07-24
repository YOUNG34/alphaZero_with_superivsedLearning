import chess
import numpy as np

import chess.svg

class Run(object):
    def __init__(self):
        self.players = [1,2]

    def current_state(self):
        square_state = np.zeros((12, 8, 8))
        board = chess.BaseBoard()
        for i in range(6):
            s = board.pieces(i + 1, True)
            for white_loc in s:
                square_state[i][7 - white_loc // 8, white_loc % 8] = 1

        for black_i in range(6):
            black_s = board.pieces(black_i + 1, False)
            for black_loc in black_s:
                square_state[black_i + 6][7 - black_loc // 8, black_loc % 8] = 1
        return square_state

    def start_self_play(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        board = chess.Board()
        states, mcts_probs, current_players = [], [], []

        while True:
            move, move_probs = player.get_action(board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.current_state())
            mcts_probs.append(move_probs)
            current_players.append(1 if board.turn == True else 2)

            # perform a move
            board.push(move)

            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                winners_z = np.zeros(len(current_players))
                winner = 1 if board.turn == True else 2
                if result != '1/2-1/2':
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()


            return result, zip(states, mcts_probs, winners_z)

    # def self_play(self):
    #     mcts_player = mcts_pure(c_puct=5, n_playout=100)
    #     board = chess.Board()
    #     while True:
    #         if board.turn == True:
    #             move = input("Please input your move:")
    #             if chess.Move.from_uci(move) not in board.legal_moves:
    #                 print("Invalid move!")
    #                 continue
    #             board.push(chess.Move.from_uci(move))
    #             print(board)
    #             print("**********************************")
    #         else:
    #             move = mcts_player.get_action(board)
    #             board.push(chess.Move.from_uci(str(move)))
    #             print(board)
    #             print("**********************************")
    #         if board.is_variant_end():
    #             if board.turn:
    #                 print("winner is : White side")
    #             else:
    #                 print("winner is : Black side")

