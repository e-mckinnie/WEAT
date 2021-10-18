import numpy as np


class WordEmbeddingTest:
    def __init__(self):
        pass

    def p_test(self):
        pass

    def tostr(self):
        pass

    # Calculate cos(x, y)
    def _cos(self, x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
