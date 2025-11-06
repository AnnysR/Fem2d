

class Material:
    def __init__(self, c_function):
        self.c_function = c_function
        
    def get_c(self, x, y):
        return self.c_function(x, y)