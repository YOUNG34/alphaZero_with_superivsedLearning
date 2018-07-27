import chess
import numpy as np
import pygame
from pygame.locals import *
import chess.svg


class Settings(object):
    """docstring for Settings"""
    def __init__(self):
        # initialize setting of game

        # screen setting
        self.screen_width = 2000
        self.screen_height = 1200
        self.bg_color = (230, 230, 230)
        self.position = [i for i in range(64)]
        self.from_position = None
        k = 0
        for i in range(7,-1,-1):
            for j in range(8):
                self.position[k] = pygame.Rect(148+j*111.75, 100+i*111.75, 111.75, 111.75)
                k+=1

class Run(object):
    def __init__(self):
        self.players = [1,2]
        self.move_uci = []

    def start_self_play(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        board = chess.Board()
        states, mcts_probs, current_players = [], [], []

        # #move_stack = ""
        # board = chess.Board()
        #
        # chess_sets = Settings()
        # screen = pygame.display.set_mode((chess_sets.screen_width, chess_sets.screen_height))
        # pygame.display.set_caption("Chess Game")
        #
        # pygame.init()
        # image_path = '/home/k1758068/Desktop/alphaGoTest-master/image/'
        # black_b = pygame.image.load(image_path + 'blackb.png').convert_alpha()
        # black_k = pygame.image.load(image_path + 'blackk.png').convert_alpha()
        # black_n = pygame.image.load(image_path + 'blackn.png').convert_alpha()
        # black_p = pygame.image.load(image_path + 'blackp.png').convert_alpha()
        # black_q = pygame.image.load(image_path + 'blackq.png').convert_alpha()
        # black_r = pygame.image.load(image_path + 'blackr.png').convert_alpha()
        #
        # white_b = pygame.image.load(image_path + 'whiteb.png').convert_alpha()
        # white_k = pygame.image.load(image_path + 'whitek.png').convert_alpha()
        # white_n = pygame.image.load(image_path + 'whiten.png').convert_alpha()
        # white_p = pygame.image.load(image_path + 'whitep.png').convert_alpha()
        # white_q = pygame.image.load(image_path + 'whiteq.png').convert_alpha()
        # white_r = pygame.image.load(image_path + 'whiter.png').convert_alpha()
        #
        # images = {3: [white_b, black_b], 6: [white_k, black_k], 2: [white_n, black_n],
        #           1: [white_p, black_p], 5: [white_q, black_q], 4: [white_r, black_r]}
        #
        # background_color = (230, 230, 230)
        # image_path = '/home/k1758068/Desktop/alphaGoTest-master/image/'
        # chess_board = pygame.image.load(image_path + 'board_image.png').convert()

        while True:
            # screen.fill(background_color)
            # chess_board_x = 100
            # chess_board_y = 50
            # # draw the board
            # screen.blit(chess_board, (chess_board_x, chess_board_y))
            #
            # d = (950 - 55) / 8
            # for i in range(64):
            #     if board.piece_at(i):
            #         piece = board.piece_at(i).piece_type
            #         color = board.piece_at(i).color
            #
            #         if color:
            #             piece = images[piece][0]
            #         else:
            #             piece = images[piece][1]
            #
            #         x = 177 + (i % 8) * d
            #         y = 23 + (8 - (i // 8)) * d
            #         screen.blit(piece, (x, y))

            move, move_probs = player.get_action(board,
                                                 temp=temp,
                                                 return_prob=1)
            if move not in self.move_uci:
                self.move_uci.append(move)
            print(self.move_uci)

            # store the data
            states.append(self.current_state())

            prob = np.zeros(1968)
            #labels_array = self.create_uci_labels()
            #label2i = {val: i for i, val in enumerate(labels_array)}
            # print(len(move_probs[0][0]))
            # print(move_probs)
            # print(move_probs[0][0])
            # for i in range(len(move_probs)):
            #     uci_move = move_probs[i][0]
            #     position = label2i[uci_move]
            #     print('position',position)
            #     prob[position] = move_probs[i][1]


            mcts_probs.append(prob)
            current_players.append(1 if board.turn == True else 2)
            # print('np.array(states).shape',np.array(states).shape)
            # print('mcts_probs',len(mcts_probs))#
            # print('current_players',len(current_players))
            # print('move',move)
            # print(board.turn)
            # print(" ************************")

            # perform a move
            board.push(chess.Move.from_uci(move))

            # draw the board
            #screen.blit(chess_board, (chess_board_x, chess_board_y))

            # d = (950 - 55) / 8
            # for i in range(64):
            #     if board.piece_at(i):
            #         piece = board.piece_at(i).piece_type
            #         color = board.piece_at(i).color
            #
            #         if color:
            #             piece = images[piece][0]
            #         else:
            #             piece = images[piece][1]
            #
            #         x = 177 + (i % 8) * d
            #         y = 23 + (8 - (i // 8)) * d
            #         screen.blit(piece, (x, y))

            # output result
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                print("result",result)
                winners_z = np.zeros(len(current_players))
                winner = 1 if board.turn == True else 2
                if result != '1/2-1/2':
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                #player.reset_player()
                return result, zip(states, mcts_probs, winners_z)
            #pygame.display.flip()

    def play_game(self, player1, player2):
        board = chess.Board()

        while True:
            if board.turn:
                move = player1.get_action(board)
            else:
                move = player2.get_action(board)
            board.push(chess.Move.from_uci(move))
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                print("result",result)
                if result != '1/2-1/2':
                    winner = 1 if board.turn == True else 2
                    print("Game end and the winner is",winner)
                else:
                    print("Game end. Tie")
                    return -1

                return winner


    def create_all_uci_labels(self):
        labels_array = []
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

        #queens_labels




    def alg_to_coord(alg):
        rank = 8 - int(alg[1])  # 0-7
        file = ord(alg[0]) - ord('a')  # 0-7
        return rank, file

    def current_state(self):
        square_state = np.zeros((12, 8, 8))
        chess_board = chess.Board()
        foo = chess_board.fen().split(' ')

        en_passant = np.zeros((8, 8), dtype=np.float32)

        if foo[3] != '-':
            eps = self.alg_to_coord(foo[3])
            en_passant[eps[0]][eps[1]] = 1

        fifty_move_count = int(foo[4])
        fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)

        castling = foo[2]
        auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
                            np.full((8, 8), int('Q' in castling), dtype=np.float32),
                            np.full((8, 8), int('k' in castling), dtype=np.float32),
                            np.full((8, 8), int('q' in castling), dtype=np.float32),
                            fifty_move,
                            en_passant]
        #ret = np.array(auxiliary_planes, dtype=np.float32)
        #assert ret.shape == (6, 8, 8)

        board = chess.BaseBoard()
        for i in range(6):
            s = board.pieces(i + 1, True)
            for white_loc in s:
                square_state[i][7 - white_loc // 8, white_loc % 8] = 1

        for black_i in range(6):
            black_s = board.pieces(black_i + 1, False)
            for black_loc in black_s:
                square_state[black_i + 6][7 - black_loc // 8, black_loc % 8] = 1

        square_state = np.array(square_state, dtype=np.float32)
        # #square_state = square_state.append([np.full((8, 8), int('K' in castling), dtype=np.float32),
        #                     np.full((8, 8), int('Q' in castling), dtype=np.float32),
        #                     np.full((8, 8), int('k' in castling), dtype=np.float32),
        #                     np.full((8, 8), int('q' in castling), dtype=np.float32),
        #                     fifty_move,
        #                     en_passant])
        state = np.vstack([square_state, auxiliary_planes])
        assert state.shape == (18,8,8)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(state)#
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        return state

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
