from __future__ import print_function

from pieces import Pieces
from termcolor import colored
    

class Board(object):

    def __init__(self):
        self.height = 8
        self.width = 8

        self.history = []
        self.pieces ={}
        self.position = list(range(self.height*self.width))
        self.key = list(range(self.height * self.width))
        value = list('x' for i in range(self.height * self.width))
        self.board_pieces = dict(zip(self.key, value))

    def init_board(self):
        # Put all the chess pieces into the board
        pieces = Pieces()
        self.pieces = pieces.return_exist_pieces()
        for key in self.pieces:
            if key in self.board_pieces:
                self.board_pieces[key] = self.pieces[key]

    def update_pieces(self, move_from, move_to):
        self.board_pieces[move_to] = self.board_pieces[move_from]
        self.board_pieces[move_from] = 'x'
        return self.board_pieces

    def return_exist_pieces(self):
        return self.board_pieces

    def graph_board(self,board_pieces):
        # Draw the board, red denotes black side, yellow denotes white side
        print("  ", "a", "b","c" , "d" ,"e" , "f" , "g" ,"h","    ")
        print()

        for i in range(self.height,-1,-1):
            if i == 0:
                continue
            print(i,end="  ")

            for j in range(self.width):
                if len(board_pieces[(i-1)*8+j]) == 2:
                    if board_pieces[(i - 1) * 8 + j][1] is 'b':
                        if j != 7:
                            print(colored(board_pieces[(i-1)*8+j][0], 'red'),end=" ")
                        if j == 7:
                            print(colored(board_pieces[(i-1)*8+j][0], 'red'), end="  ")
                    else:
                        if j != 7:
                            print(colored(board_pieces[(i-1)*8+j][0], 'yellow'),end=" ")
                        if j == 7:
                            print(colored(board_pieces[(i-1)*8+j][0], 'yellow'), end="  ")
                else:
                    if j != 7:
                        print(board_pieces[(i - 1) * 8 + j][0], end=" ")
                    if j == 7:
                        print(board_pieces[(i - 1) * 8 + j][0], end="  ")

            print(i)

        print()
        print("  ", "a", "b","c" , "d" ,"e" , "f" , "g" ,"h","    ")

    def loc_to_move(self, location):
        move = 0
        column = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        x = location[0]
        y = int(location[1])
        move = (y-1) * 8 + column[x]
        return move

    def move_to_loc(self, move):
        re_column = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h'}
        x = re_column[move % 8]
        y = (move // 8) + 1
        return x+str(y)


class Play(object):

    def run(self):
        '''run the game'''
        board = Board()
        board.init_board()
        pieces = board.return_exist_pieces()
        board.graph_board(pieces)
        while True:
        #input_loc = input("Please choose the side you like to play as (w or b): ")
            input_loc = input("Please enter the coordinate of pieces:")

            chosen_piece, from_loc, to_loc = input_loc.split(",")

            chosen_piece_loc= board.loc_to_move(from_loc)

            target_loc= board.loc_to_move(to_loc)

            pieces = board.update_pieces(chosen_piece_loc,target_loc)
            board.graph_board(pieces)


if __name__ == '__main__':

    play = Play()
    play.run()

