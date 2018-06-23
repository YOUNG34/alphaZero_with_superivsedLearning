class Bishop(object):
    def __init__(self):
        self.init_location = {2:['B','w'],5:['B','w'],58:['B','b'],61:['B','b']}

    def pieces(self):
        return self.init_location