import chess
import numpy as np
import pygame
from pygame.locals import *
import chess.svg

import os
import chess.pgn
from time import time
import json
import collections
from collections import OrderedDict

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
            #print(self.move_uci)

            # store the data
            states.append(self.current_state())
            labels_array = self.create_all_uci_labels()
            prob = np.zeros(len(labels_array))
            #print(len(labels_array))
            label2i = {val: i for i, val in enumerate(labels_array)}
            # print(len(move_probs[0][0]))
            # print(move_probs)
            # print(move_probs[0][0])
            for i in range(len(move_probs)):
                uci_move = move_probs[i][0]
                position = label2i[uci_move]
                #print('position',position)
                prob[position] = move_probs[i][1]


            mcts_probs.append(prob)
            current_players.append(1 if board.turn == True else 2)
            # print('np.array(states).shape',np.array(states).shape)
            # print('mcts_probs',len(mcts_probs))#
            # print('current_players',len(current_players))
            #print('move',move)
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


    def start_supervised_learning(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        board = chess.Board()
        states, mcts_probs, current_players = [], [], []

        while True:

            move, move_probs = player.get_action(board,
                                                 temp=temp,
                                                 return_prob=1)
            if move not in self.move_uci:
                self.move_uci.append(move)

            # store the data
            states.append(self.current_state())
            labels_array = self.create_all_uci_labels()
            prob = np.zeros(len(labels_array))

            label2i = {val: i for i, val in enumerate(labels_array)}

            for i in range(len(move_probs)):
                uci_move = move_probs[i][0]
                position = label2i[uci_move]

                prob[position] = move_probs[i][1]


            mcts_probs.append(prob)
            current_players.append(1 if board.turn == True else 2)

            # perform a move
            board.push(chess.Move.from_uci(move))

            # output result
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                print("result",result)
                winners_z = np.zeros(len(current_players))
                winner = 1 if board.turn == True else 2
                if result != '1/2-1/2':
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                return result, zip(states, mcts_probs, winners_z)




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
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        direction_for_queen_and_king = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        direction_for_knight = [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]

        for l1 in range(8):
            for n1 in range(8):
                for d in range(8):
                    for i in range(8):
                        (l2, n2) = (l1 + direction_for_queen_and_king[d][0] * i,
                                    n1 + direction_for_queen_and_king[d][1] * i)
                        (l3, n3) = (l1 + direction_for_knight[d][0] * i,
                                    n1 + direction_for_knight[d][1] * i)
                        if l2 in range(8) and n2 in range(8):
                            move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                            labels_array.append(move)
                        if l3 in range(8) and n3 in range(8):
                            move = letters[l1] + numbers[n1] + letters[l3] + numbers[n3]
                            labels_array.append(move)
        pawn = ['a2a3','a2a4','b2b3','b2b4','c2c3','c2c4','d2d3','d2d4','e2e3','e2e4','f2f3','f2f4','g2g3','g2g4',
                'h2h3','h2h4','a7a6','a7a5','b7b6','b7b5','c7c6','c7c5','d7d6','d7d5','e7e6','e7e5','f7f6','f7f5',
                'g7g6','g7g5','h7h6','h7h5',
                'a7a8n','a7a8b','a7a8r','a7a8q','a7b8n','a7b8b','a7b8r','a7b8q',
                'b7a8n','b7a8b','b7a8r','b7a8q','b7b8n','b7b8b','b7b8r','b7b8q','b7c8n','b7c8b','b7c8r','b7c8q'
                ,'c7b8n','c7b8b','c7b8r','c7b8q','c7c8n','c7c8b','c7c8r','c7c8q','c7d8n','c7d8b','c7d8r','c7d8q'
                ,'d7c8n','d7c8b','d7c8r','d7c8q','d7d8n','d7d8b','d7d8r','d7d8q','d7e8n','d7e8b','d7e8r','d7e8q'
                ,'e7d8n','e7e8n','e7f8n','e7d8b','e7e8b','e7f8b','e7d8r','e7e8r','e7f8r','e7d8q','e7e8q','e7f8q'
                ,'f7e8n','f7f8n','f7g8n','f7e8b','f7f8b','f7g8b','f7e8r','f7f8r','f7g8r','f7e8q','f7f8q','f7g8q'
                ,'g7f8n','g7g8n','g7h8n','g7f8b','g7g8b','g7h8b','g7f8r','g7g8r','g7h8r','g7f8q','g7g8q','g7h8q'
                ,'h7g8n','h7h8n','h7g8b','h7h8b','h7g8r','h7h8r','h7g8q','h7h8q'
                ,'a2a1n','a2b1n','a2a1b','a2b1b','a2a1r','a2b1r','a2a1q','a2b1q'
                ,'b2a1n','b2b1n','b2c1n','b2a1b','b2b1b','b2c1b','b2a1r','b2b1r','b2c1r','b2a1q','b2b1q','b2c1q'
                ,'c2b1n','c2c1n','c2d1n','c2b1b','c2c1b','c2d1b','c2b1r','c2c1r','c2d1r','c2b1q','c2c1q','c2d1q'
                ,'d2c1n','d2d1n','d2e1n','d2c1b','d2d1b','d2e1b','d2c1r','d2d1r','d2e1r','d2c1q','d2d1q','d2e1q'
                ,'e2d1n','e2e1n','e2f1n','e2d1b','e2e1b','e2f1b','e2d1r','e2e1r','e2f1r','e2d1q','e2e1q','e2f1q'
                ,'f2e1n','f2f1n','f2g1n','f2e1b','f2f1b','f2g1b','f2e1r','f2f1r','f2g1r','f2e1q','f2f1q','f2g1q'
                ,'g2f1n','g2g1n','g2h1n','g2f1b','g2g1b','g2h1b','g2f1r','g2g1r','g2h1r','g2f1q','g2g1q','g2h1q'
                ,'h2g1n','h2h1n','h2g1b','h2h1b','h2g1r','h2h1r','h2g1q','h2h1q',
                'e1g1','e1c1','e8c8','e8g8'
                ]

        for i in range(len(pawn)):
            labels_array.append(pawn[i])

        return labels_array





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

        state = np.vstack([square_state, auxiliary_planes])
        assert state.shape == (18,8,8)

        return state


class Supervised_learning(object):

    def __init__(self):
        self.min_elo_policy = 500
        self.max_elo_policy = 1800
        self.buff = []
        run = Run()
        self.labels = run.create_all_uci_labels()

    def get_games_from_file(self):
        #file = open('/Users/zeyang/Desktop/alphaGoTest-master/supervised_learning_data/ficsgamesdb_201801_chess_nomovetimes_10994.pgn'
        #            , errors='ignore')
        file = open('/home/k1758068/Desktop/alphaGoTest-master/supervised_learning_data/ficsgamesdb_201701_chess_nomovetimes_10312.pgn')

        games = [[]]*1000

        for i in range(len(games)):
            games[i] = chess.pgn.read_game(file)

        return games

    def clip_elo_policy(self, elo):
        return min(1, max(0, elo - self.min_elo_policy) / self.max_elo_policy)

    def get_data(self,game):
        result = game.headers["Result"]
        white_elo, black_elo = int(game.headers["WhiteElo"]),\
                               int(game.headers["BlackElo"])
        white_weight = self.clip_elo_policy(white_elo)
        black_weight = self.clip_elo_policy(black_elo)

        moves = []
        i = 0
        while not game.is_end():

            game = game.variation(0)
            moves.append(game.move.uci())
            i += 1
        moves.append([result,white_weight,black_weight])

        return moves

    def save_data(self, moves,id):
        self.file_name = '/home/k1758068/Desktop/alphaGoTest-master/move_json_files/move'+id+'.json'
        #print(moves)
        # moves = np.array(moves)
        # np.savetxt('/Users/zeyang/Desktop/alphaGoTest-master/moves.txt',moves)
        # with open(file_name, 'a') as f:
        #     f.write(str(moves))
        with open(self.file_name, "w") as f:
            json.dump(moves, f)

    def prepare(self):
        games = self.get_games_from_file()
        i = 0
        for game in games:
            i += 1
            a = self.get_data(game)
            # for i in range(len(a)):
            # print(a)
            self.save_data(a, str(i))

    def supervised_learning_run(self,index):
        print('read file:',index)
        run = Run()
        board = chess.Board()
        with open('/home/k1758068/Desktop/alphaGoTest-master/move_json_files/move'+str(index)+'.json', "rt") as f:
            self.buff = json.load(f)
            result = self.buff[-1][0]
            white_weight = self.buff[-1][1]
            black_weight = self.buff[-1][2]
            result_real = 0
            states, mcts_probs, current_players = [], [], []

            for j in range(len(self.buff)-1):

                move = self.buff[j] #结尾记得加 j += 1
                # store the data
                states.append(run.current_state())
                prob = np.zeros(len(self.labels))

                # print(len(labels_array))
                label2i = {val: i for i, val in enumerate(self.labels)}
                # print(len(move_probs[0][0]))
                # print(move_probs)
                # print(move_probs[0][0])
                position = label2i[move]
                # print('position',position)
                if board.turn == True:
                    prob[position] = white_weight
                else:
                    prob[position] = black_weight
                mcts_probs.append(prob)
                current_players.append(1 if board.turn == True else 2)

                # perform a move
                board.push(chess.Move.from_uci(move))
                #print("game over:",board.is_game_over(claim_draw=True))

            result_real = board.result(claim_draw=True)
            print("result",result,result_real)
            winners_z = np.zeros(len(current_players))
            winner = 1 if board.turn == False else 2
            if result != '1/2-1/2':
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            return result, zip(states, mcts_probs, winners_z)
import chess
import numpy as np
import pygame
from pygame.locals import *
import chess.svg

import os
import chess.pgn
from time import time
import json
import collections
from collections import OrderedDict


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

            #print(self.move_uci)

            # store the data
            states.append(self.current_state())
            labels_array = self.create_all_uci_labels()
            prob = np.zeros(len(labels_array))
            #print(len(labels_array))
            label2i = {val: i for i, val in enumerate(labels_array)}
            # print(len(move_probs[0][0]))
            # print(move_probs)
            # print(move_probs[0][0])
            for i in range(len(move_probs)):
                uci_move = move_probs[i][0]
                position = label2i[uci_move]
                #print('position',position)
                prob[position] = move_probs[i][1]


            mcts_probs.append(prob)
            current_players.append(1 if board.turn == True else 2)

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
        numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
        direction_for_queen_and_king = [(0, 1), (-1, 1), (-1, 0), (-1, -1), 
                                        (0, -1), (1, -1), (1, 0), (1, 1)]
        direction_for_knight = [(-2, -1), (-1, -2), (-2, 1), (1, -2), 
                                (2, -1), (-1, 2), (2, 1), (1, 2)]

        for l1 in range(8):
            for n1 in range(8):
                for d in range(8):
                    for i in range(8):
                        (l2, n2) = (l1 + direction_for_queen_and_king[d][0] * i,
                                    n1 + direction_for_queen_and_king[d][1] * i)
                        (l3, n3) = (l1 + direction_for_knight[d][0] * i, 
                                    n1 + direction_for_knight[d][1] * i)
                        if l2 in range(8) and n2 in range(8):
                            move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                            labels_array.append(move)
                        if l3 in range(8) and n3 in range(8):
                            move = letters[l1] + numbers[n1] + letters[l3] + numbers[n3]
                            labels_array.append(move)
        pawn = ['a2a3','a2a4','b2b3','b2b4','c2c3','c2c4','d2d3','d2d4','e2e3','e2e4','f2f3','f2f4','g2g3','g2g4',
                'h2h3','h2h4','a7a6','a7a5','b7b6','b7b5','c7c6','c7c5','d7d6','d7d5','e7e6','e7e5','f7f6','f7f5',
                'g7g6','g7g5','h7h6','h7h5',
                'a7a8n','a7a8b','a7a8r','a7a8q','a7b8n','a7b8b','a7b8r','a7b8q',
                'b7a8n','b7a8b','b7a8r','b7a8q','b7b8n','b7b8b','b7b8r','b7b8q','b7c8n','b7c8b','b7c8r','b7c8q'
                ,'c7b8n','c7b8b','c7b8r','c7b8q','c7c8n','c7c8b','c7c8r','c7c8q','c7d8n','c7d8b','c7d8r','c7d8q'
                ,'d7c8n','d7c8b','d7c8r','d7c8q','d7d8n','d7d8b','d7d8r','d7d8q','d7e8n','d7e8b','d7e8r','d7e8q'
                ,'e7d8n','e7e8n','d7f8n','e7d8b','e7e8b','d7f8b','e7d8r','e7e8r','d7f8r','e7d8q','e7e8q','d7f8q'
                ,'f7e8n','f7f8n','f7g8n','f7e8b','f7f8b','f7g8b','f7e8r','f7f8r','f7g8r','f7e8q','f7f8q','f7g8q'
                ,'g7f8n','g7g8n','g7h8n','g7f8b','g7g8b','g7h8b','g7f8r','g7g8r','g7h8r','g7f8q','g7g8q','g7h8q'
                ,'h7g8n','h7h8n','h7g8b','h7h8b','h7g8r','h7h8r','h7g8q','h7h8q'
                ,'a2a1n','a2b1n','a2a1b','a2b1b','a2a1r','a2b1r','a2a1q','a2b1q'
                ,'b2a1n','b2b1n','b2c1n','b2a1b','b2b1b','b2c1b','b2a1r','b2b1r','b2c1r','b2a1q','b2b1q','b2c1q'
                ,'c2b1n','c2c1n','c2d1n','c2b1b','c2c1b','c2d1b','c2b1r','c2c1r','c2d1r','c2b1q','c2c1q','c2d1q'
                ,'d2c1n','d2d1n','d2e1n','d2c1b','d2d1b','d2e1b','d2c1r','d2d1r','d2e1r','d2c1q','d2d1q','d2e1q'
                ,'e2d1n','e2e1n','e2f1n','e2d1b','e2e1b','e2f1b','e2d1r','e2e1r','e2f1r','e2d1q','e2e1q','e2f1q'
                ,'f2e1n','f2f1n','f2g1n','f2e1b','f2f1b','f2g1b','f2e1r','f2f1r','f2g1r','f2e1q','f2f1q','f2g1q'
                ,'g2f1n','g2g1n','g2h1n','g2f1b','g2g1b','g2h1b','g2f1r','g2g1r','g2h1r','g2f1q','g2g1q','g2h1q'
                ,'h2g1n','h2h1n','h2g1b','h2h1b','h2g1r','h2h1r','h2g1q','h2h1q',
                'e1g1','e1c1','e8c8','e8g8'
                ]

        for i in range(len(pawn)):
            labels_array.append(pawn[i])

        return labels_array



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

class Supervised_learning(object):

    def __init__(self):
        self.min_elo_policy = 500
        self.max_elo_policy = 1800
        self.buff = []
        run = Run()
        self.labels = run.create_all_uci_labels()

    def get_games_from_file(self):
        file = open('/Users/zeyang/Desktop/alphaGoTest-master/supervised_learning_data/ficsgamesdb_201801_chess_nomovetimes_10994.pgn'
                    , errors='ignore')

        games = [[]]*51

        for i in range(len(games)):
            games[i] = chess.pgn.read_game(file)

        return games

    def clip_elo_policy(self, elo):
        return min(1, max(0, elo - self.min_elo_policy) / self.max_elo_policy)

    def get_data(self,game):
        result = game.headers["Result"]
        white_elo, black_elo = int(game.headers["WhiteElo"]),\
                               int(game.headers["BlackElo"])
        white_weight = self.clip_elo_policy(white_elo)
        black_weight = self.clip_elo_policy(black_elo)

        moves = []
        i = 0
        while not game.is_end():

            game = game.variation(0)
            moves.append(game.move.uci())
            i += 1
        moves.append([result,white_weight,black_weight])

        return moves

    def save_data(self, moves,id):
        self.file_name = '/Users/zeyang/Desktop/alphaGoTest-master/move_json_files/move'+id+'.json'
        #print(moves)
        # moves = np.array(moves)
        # np.savetxt('/Users/zeyang/Desktop/alphaGoTest-master/moves.txt',moves)
        # with open(file_name, 'a') as f:
        #     f.write(str(moves))
        with open(self.file_name, "w") as f:
            json.dump(moves, f)

    def prepare(self):
        games = self.get_games_from_file()
        i = 0
        for game in games:
            i += 1
            a = self.get_data(game)
            # for i in range(len(a)):
            # print(a)
            self.save_data(a, str(i))

    def supervised_learning_run(self,index):
        print('read file:',index)
        run = Run()
        board = chess.Board()
        with open('/Users/zeyang/Desktop/alphaGoTest-master/move_json_files/move'+str(index)+'.json', "rt") as f:
            self.buff = json.load(f)
            result = self.buff[-1][0]
            white_weight = self.buff[-1][1]
            black_weight = self.buff[-1][2]
            result_real = 0
            states, mcts_probs, current_players = [], [], []

            for j in range(len(self.buff)-1):

                move = self.buff[j] #结尾记得加 j += 1
                # store the data
                states.append(run.current_state())
                prob = np.zeros(len(self.labels))

                # print(len(labels_array))
                label2i = {val: i for i, val in enumerate(self.labels)}
                # print(len(move_probs[0][0]))
                # print(move_probs)
                # print(move_probs[0][0])
                position = label2i[move]
                # print('position',position)
                if board.turn == True:
                    prob[position] = white_weight
                else:
                    prob[position] = black_weight
                mcts_probs.append(prob)
                current_players.append(1 if board.turn == True else 2)

                # perform a move
                board.push(chess.Move.from_uci(move))
                print("game over:",board.is_game_over(claim_draw=True))

            result_real = board.result(claim_draw=True)
            print("result",result,result_real)
            winners_z = np.zeros(len(current_players))
            winner = 1 if board.turn == False else 2
            if result != '1/2-1/2':
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            #player.reset_player()
            print(type(result))
            return result, zip(states, mcts_probs, winners_z)





