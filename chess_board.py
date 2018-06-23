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
        self.empty_board = dict(zip(self.key, value))

    def graph_board(self):



        # Put exist pieces into the board
        pieces = Pieces()
        self.pieces = pieces.return_exist_pieces()
        for key in self.pieces:
            if key in self.empty_board:
                self.empty_board[key] = self.pieces[key]

        # Draw the board, red denotes black side, yellow denotes white side
        print("  ", "a", "b","c" , "d" ,"e" , "f" , "g" ,"h","    ")
        print()

        for i in range(self.height,-1,-1):
            if i == 0:
                continue
            print(i,end="  ")

            for j in range(self.width):
                if len(self.empty_board[(i-1)*8+j]) == 2:
                    if self.empty_board[(i - 1) * 8 + j][1] is 'b':
                        if j != 7:
                            print(colored(self.empty_board[(i-1)*8+j][0], 'red'),end=" ")
                        if j == 7:
                            print(colored(self.empty_board[(i-1)*8+j][0], 'red'), end="  ")
                    else:
                        if j != 7:
                            print(colored(self.empty_board[(i-1)*8+j][0], 'yellow'),end=" ")
                        if j == 7:
                            print(colored(self.empty_board[(i-1)*8+j][0], 'yellow'), end="  ")
                else:
                    if j != 7:
                        print(self.empty_board[(i - 1) * 8 + j][0], end=" ")
                    if j == 7:
                        print(self.empty_board[(i - 1) * 8 + j][0], end="  ")

            print(i)

        print()
        print("  ", "a", "b","c" , "d" ,"e" , "f" , "g" ,"h","    ")



# for i in range(self.height,-1,-1):
#     if i==0:
#         continue
#     print(i,end="  ")
#     if i == 1:
#         print('R', 'N', 'B', 'Q', 'K', 'B','N','R',end="  ")
#     elif i == 2:
#         print('P','P','P','P','P','P','P','P',end="  ")
#     elif i == 7:
#         print('P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',end="  ")
#     elif i == 8:
#         print('R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R',end="  ")
#     else:
#         for j in range(self.width):
#             k += 1
#             if k == 8:
#                 print(0,end="  ")
#                 k = 0
#             else:
#                 print(0,end = " ")
#     print(i)