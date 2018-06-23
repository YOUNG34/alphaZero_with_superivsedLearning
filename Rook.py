
class Rook(object):
    def __init__(self):
        self.init_location = {0:['R','w'],7:['R','w'],63:['R','b'],56:['R','b']}

    def pieces(self):
        return self.init_location
