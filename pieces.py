from chess_pieces.Rook import Rook
from chess_pieces.Knight import Knight
from chess_pieces.Bishop import Bishop
from chess_pieces.Queen import Queen
from chess_pieces.King import King
from chess_pieces.Pawn import Pawn

from collections import ChainMap

class Pieces(object):
    def return_exist_pieces(self):
        rook = Rook().pieces()
        bishop = Bishop().pieces()
        knight = Knight().pieces()
        queen = Queen().pieces()
        king = King().pieces()
        pawn = Pawn().pieces()

        self.pieces = ChainMap(rook, bishop, knight, queen, king, pawn)
        return self.pieces
