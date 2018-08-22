import sys
import pygame
import chess
from mcts_pure import MCTSPlayer as mcts_pure
from pygame.locals import *
from tkinter import *
import pop_dialog
import json
import chess.pgn
import time

class Settings(object):
    """docstring for Settings"""
    def __init__(self):
        # initialize setting of game

        # screen setting
        self.screen_width = 700
        self.screen_height = 700
        self.bg_color = (230, 230, 230)
        self.position = [i for i in range(64)]
        self.from_position = None
        k = 0
        for i in range(7,-1,-1):
            for j in range(8):
                self.position[k] = pygame.Rect(38+j*78.125, 40+i*78.125, 78.125, 78.125)
                k+=1



class GUI(object):


    def is_chess_clicked(self,position, event):

        for each in position:
            if (each.collidepoint(event.pos)):
                return position.index(each)
        return None


    def run_game(self):
        mcts_player = mcts_pure(c_puct=5, n_playout=10)
        move_stack = ""
        board = chess.Board()

        chess_sets = Settings()
        screen = pygame.display.set_mode((chess_sets.screen_width, chess_sets.screen_height))
        pygame.display.set_caption("Chess Game")

        pygame.init()
        image_path = '/home/k1758068/Desktop/alphaGoTest-master/image/'
        black_b = pygame.image.load(image_path + 'blackb.png').convert_alpha()
        black_k = pygame.image.load(image_path + 'blackk.png').convert_alpha()
        black_n = pygame.image.load(image_path + 'blackn.png').convert_alpha()
        black_p = pygame.image.load(image_path + 'blackp.png').convert_alpha()
        black_q = pygame.image.load(image_path + 'blackq.png').convert_alpha()
        black_r = pygame.image.load(image_path + 'blackr.png').convert_alpha()

        white_b = pygame.image.load(image_path + 'whiteb.png').convert_alpha()
        white_k = pygame.image.load(image_path + 'whitek.png').convert_alpha()
        white_n = pygame.image.load(image_path + 'whiten.png').convert_alpha()
        white_p = pygame.image.load(image_path + 'whitep.png').convert_alpha()
        white_q = pygame.image.load(image_path + 'whiteq.png').convert_alpha()
        white_r = pygame.image.load(image_path + 'whiter.png').convert_alpha()

        images = {3: [white_b, black_b], 6: [white_k, black_k], 2: [white_n, black_n],
                       1: [white_p, black_p], 5: [white_q, black_q], 4: [white_r, black_r]}

        background_color = (230,230,230)
        image_path = '/home/k1758068/Desktop/alphaGoTest-master/image/'
        chess_board = pygame.image.load(image_path + 'board_image.png').convert()


        with open('/home/k1758068/Desktop/alphaGoTest-master/move_json_files/move12136.json', "rt") as f:
            screen.fill(background_color)
            chess_board_x = 0
            chess_board_y = 0


            screen.blit(chess_board,(chess_board_x,chess_board_y))
            d = (700-80)/8
            for i in range(64):
                if board.piece_at(i):
                    piece = board.piece_at(i).piece_type
                    color = board.piece_at(i).color

                    if color:
                        piece = images[piece][0]
                    else:
                        piece = images[piece][1]

                    x = 48 + (i % 8) * d
                    y = (8 - (i // 8))*d - 26
                    screen.blit(piece, (x, y))

            self.buff = json.load(f)
            result = self.buff[-1][0]
            white_weight = self.buff[-1][1]
            black_weight = self.buff[-1][2]
            result_real = 0
            states, mcts_probs, current_players = [], [], []
            print('stsrt')
            for j in range(len(self.buff)-1):
                screen.fill(background_color)
                chess_board_x = 100
                chess_board_y = 50

                screen.blit(chess_board, (chess_board_x, chess_board_y))

                d = (950 - 55) / 8
                for i in range(64):
                    if board.piece_at(i):
                        piece = board.piece_at(i).piece_type
                        color = board.piece_at(i).color

                        screen.fill(background_color)
                        chess_board_x = 0
                        chess_board_y = 0

                        screen.blit(chess_board, (chess_board_x, chess_board_y))

                        d = (700 - 80) / 8
                        for i in range(64):
                            if board.piece_at(i):
                                piece = board.piece_at(i).piece_type
                                color = board.piece_at(i).color

                                if color:
                                    piece = images[piece][0]
                                else:
                                    piece = images[piece][1]

                                x = 48 + (i % 8) * d
                                y = (8 - (i // 8)) * d - 26
                                screen.blit(piece, (x, y))
                myfont = pygame.font.SysFont('test', 30)
                textsurface = myfont.render(move_stack, True, (0, 0, 0))
                screen.blit(textsurface, (1000, 1000))

                move = self.buff[j]
                # perform a move
                board.push(chess.Move.from_uci(move))
                screen.fill(background_color)

                chess_board_x = 0
                chess_board_y = 0

                screen.blit(chess_board, (chess_board_x, chess_board_y))

                d = (700 - 80) / 8
                for i in range(64):
                    if board.piece_at(i):
                        piece = board.piece_at(i).piece_type
                        color = board.piece_at(i).color

                        if color:
                            piece = images[piece][0]
                        else:
                            piece = images[piece][1]

                        x = 48 + (i % 8) * d
                        y = (8 - (i // 8)) * d - 26
                        screen.blit(piece, (x, y))
                time.sleep(0.5)
                #print("game over:",board.is_game_over(claim_draw=True))
                pygame.display.flip()
                result_real = board.result(claim_draw=True)




            pygame.display.flip()




if __name__ == '__main__':
    run = GUI()
    run.run_game()

