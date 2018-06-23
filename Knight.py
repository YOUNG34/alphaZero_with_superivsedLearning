class Knight(object):
    def __init__(self):
        self.init_location = {1:['N','w'],6:['N','w'],57:['N','b'],62:['N','b']}

    def pieces(self):
        return self.init_location