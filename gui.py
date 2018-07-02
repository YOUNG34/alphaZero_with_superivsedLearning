import sys
import pygame
import chess
from mcts_pure import MCTSPlayer as mcts_pure
from pygame.locals import *

class Settings(object):
    """docstring for Settings"""
    def __init__(self):
        # initialize setting of game

        # screen setting
        self.screen_width = 1200
        self.screen_height = 1200
        self.bg_color = (230, 230, 230)
        self.position = [i for i in range(64)]
        self.from_position = None
        k = 0
        for i in range(7,-1,-1):
            for j in range(8):
                self.position[k] = pygame.Rect(148+j*111.75, 100+i*111.75, 111.75, 111.75)
                k+=1



def is_chess_clicked(position, event):
    for each in position:
        if (each.collidepoint(event.pos)):
            return position.index(each)
    return None


def run_game():
    mcts_player = mcts_pure(c_puct=5, n_playout=1)
    board = chess.Board()
    pygame.init()
    chess_sets = Settings()
    screen = pygame.display.set_mode((chess_sets.screen_width, chess_sets.screen_height))
    pygame.display.set_caption("Chess game")

    background_color = (230,230,230)
    image_path = '/home/k1758068/Desktop/alphaGoTest-master/image/'
    chess_board = pygame.image.load(image_path + 'board_image.png').convert()

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

    images = {3:[white_b, black_b], 6:[white_k, black_k], 2:[white_n, black_n],
              1:[white_p, black_p], 5:[white_q, black_q], 4:[white_r, black_r]}

    while True:

        screen.fill(background_color)
        chess_board_x = 100
        chess_board_y = 50


        screen.blit(chess_board,(chess_board_x,chess_board_y))

        d = (950-55)/8
        #piece_x, piece_y = 55+(j-1)*d, 55+(i-1)*d

        for i in range(64):
            if board.piece_at(i):
                piece = board.piece_at(i).piece_type
                color = board.piece_at(i).color
                #print(piece,i)
                if color:
                    piece = images[piece][0]
                else:
                    piece = images[piece][1]

                x = 177 + (i % 8) * d
                y = 23 + (8 - (i // 8))*d
                screen.blit(piece, (x, y))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if board.turn == True:
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        selected_position = is_chess_clicked(chess_sets.position, event)
                        select_piece = board.piece_at(selected_position)

                        if select_piece.color == True:
                            from_position = selected_position
                            print(from_position)
                        else:
                            pass

                    if event.button == 3:
                        selected_position = is_chess_clicked(chess_sets.position, event)
                        to_position = selected_position
                        move = chess.Move(from_position, to_position)
                        if move not in board.legal_moves:
                            print("invalide move!")
                        else:
                            board.push(chess.Move(from_position, to_position))

            else:
                move = mcts_player.get_action(board)
                board.push(chess.Move.from_uci(str(move)))

        pygame.display.flip()


if __name__ == '__main__':
    run_game()
