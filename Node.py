class Node:
    def __init__(self, board):
        self.N = 0
        self.V = 0
        self.Child_nodes = {}
        self.board = board

    def update(self,r):
        self.V += r
        self.N += 1

    def Q(self, model):
        c_puct = 0.2    #hyperparameter
        P = model.value(self.board)

        Q = (self.V + P)/(self.N + 1)
        return Q
