import sys
import pygame
import chess
from mcts_for_train import MCTSPlayer as mcts_trainded_player
from policy_value_net import Policy_value_net
from mcts_pure import MCTSPlayer as mcts_player
from pygame.locals import *
from tkinter import *
import pop_dialog


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
        self.policy_value_net = Policy_value_net(model_file='./current_policy.model')
        trained_player = mcts_trainded_player(self.policy_value_net.policy_value_fn,c_puct=5, n_playout=10)
        pure_player = mcts_player(c_puct=5, n_playout=10)

        move_stack = ""
        board = chess.Board()

        chess_sets = Settings()
        screen = pygame.display.set_mode((chess_sets.screen_width, chess_sets.screen_height))
        pygame.display.set_caption("Chess Game")

        pygame.init()
        image_path = '/home/k1758068/Desktop/final_project/image/'
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
        chess_board = pygame.image.load(image_path + 'board_image.png').convert()



        while True:

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
            # myfont = pygame.font.SysFont('test', 30)
            # textsurface = myfont.render(move_stack, True, (0, 0, 0))
            # screen.blit(textsurface, (1000, 1000))
            #try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # if board.turn == True:
            #     move = trained_player.get_action(board)
            #     board.push(chess.Move.from_uci(str(move)))
            #     move_stack += "," + board.peek().uci()
            #     print(move_stack)
            # else:
            #     move = pure_player.get_action(board)
            #     board.push(chess.Move.from_uci(str(move)))
            #     move_stack += "," + board.peek().uci()
            #     print(move_stack)
                if event.type == MOUSEBUTTONDOWN:
                    print(event)
                    if board.turn == True:
                        if event.button == 1:
                            selected_position = self.is_chess_clicked(chess_sets.position, event)
                            select_piece = board.piece_at(selected_position)

                            if select_piece.color == True or select_piece.color != True:
                                from_position = selected_position
                                print(from_position)
                            else:
                                pass


                        if event.button == 3:
                            selected_position = self.is_chess_clicked(chess_sets.position, event)
                            to_position = selected_position
                            a = [48,49,50,51,52,53,54,55]
                            # The promotion precess of PAWN pieces
                            if from_position in a and select_piece.piece_type == 1:
                                #a = self.create_dialog()#
                                a = pop_dialog.main()
                                move = chess.Move(from_position, to_position, promotion = int(a), drop=None)
                                #print(int(a),move)
                            else:
                                move = chess.Move(from_position, to_position)

                            # Move check and make move
                            if move not in board.legal_moves:
                                print("invalide move!",move,board.legal_moves)
                            else:
                                board.push(move)
                                move_stack += "," + board.peek().uci()
                                print(move_stack)
                        myfont = pygame.font.SysFont('test',30)

                        textsurface = myfont.render(move_stack,True,(0,0,0))
                        screen.blit(textsurface,(100,100))
                    else:

                        if event.button == 1:
                            selected_position = self.is_chess_clicked(chess_sets.position, event)
                            select_piece = board.piece_at(selected_position)

                            if select_piece.color == False:
                                from_position = selected_position
                                print(from_position)
                            else:
                                pass

                        if event.button == 3:
                            selected_position = self.is_chess_clicked(chess_sets.position, event)
                            to_position = selected_position
                            a = [8, 9, 10, 11, 12, 13, 14, 15]
                            # The promotion precess of PAWN pieces
                            if from_position in a and select_piece.piece_type == 1:
                                # a = self.create_dialog()#
                                a = pop_dialog.main()
                                move = chess.Move(from_position, to_position, promotion=int(a), drop=None)
                                print(int(a), move)
                            else:
                                move = chess.Move(from_position, to_position)

                            # Move check and make move
                            if move not in board.legal_moves:
                                print("invalide move!", move, board.legal_moves)
                            else:
                                board.push(move)
                                move_stack += "," + board.peek().uci()
                                myfont = pygame.font.SysFont('test', 30)

                                textsurface = myfont.render(move_stack, True, (0, 0, 0))
                                screen.blit(textsurface, (100, 100))


            #
            # except:
            #    print("Error")
            #    continue

            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                print(result)
                break


            pygame.display.flip()


if __name__ == '__main__':
    run = GUI()
    run.run_game()

