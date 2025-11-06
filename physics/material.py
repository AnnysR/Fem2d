

class Material:
    def __init__(self, cFunction):
        self.cFunction = cFunction

    def getC(self, x, y):
        return self.cFunction(x, y)