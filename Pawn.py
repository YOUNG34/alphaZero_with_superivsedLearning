class Pawn(object):
    def __init__(self):
        self.init_location = {8:['P','w'],9:['P','w'],10:['P','w'],11:['P','w'],12:['P','W'],13:['P','W'],14:['P','W'],15:['P','W'],
                              48:['P','b'],49:['P','b'],50:['P','b'],51:['P','b'],52:['P','b'],53:['P','b'],54:['P','b'],55:['P','b']}

    def pieces(self):
        return self.init_location