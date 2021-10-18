import numpy as np

from wordembeddingtest import WordEmbeddingTest


class WEFAT(WordEmbeddingTest):
    def __init__(self, W, A, B):
        self.W = W
        self.A = A
        self.B = B

    # Return all calculated s
    def all_s(self):
        statistics = {}
        for w in self.W:
            statistic = self.s(self.W[w])
            statistics[w] = statistic
        return statistics

    # Calculate s(w, A, B)
    def s(self, w):
        a_cos = np.array([self._cos(w, a) for a in self.A])
        b_cos = np.array([self._cos(w, b) for b in self.B])

        return (np.mean(a_cos) - np.mean(b_cos)) / np.std(np.concatenate((a_cos, b_cos)))

    def tostr(self):
        return "WEFAT"
