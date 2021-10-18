import numpy as np

from wordembeddingtest import WordEmbeddingTest


class WEAT(WordEmbeddingTest):
    def __init__(self, X, Y, A, B):
        self.X = X
        self.Y = Y
        self.A = A
        self.B = B

    # Calculate effect size
    def effect_size(self):
        x_s = np.array([self.s(x) for x in self.X])
        y_s = np.array([self.s(y) for y in self.Y])

        return (np.mean(x_s) - np.mean(y_s)) / np.std(np.concatenate((x_s, y_s)))

    # Calculate s(w, A, B)
    def s(self, w):
        a_cos = np.array([self._cos(w, a) for a in self.A])
        b_cos = np.array([self._cos(w, b) for b in self.B])

        return (np.mean(a_cos) - np.mean(b_cos))

    # Get name of test
    def tostr(self):
        return "WEAT"

    # Get p-value
    # {(X_i, Y_i)}_i is all partitions of X union Y into two sets of equal size
    # p-value = Pr_i[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    def p_value(self):
        x_s = np.array([self.s(x) for x in self.X])
        y_s = np.array([self.s(y) for y in self.Y])

        observed_s = np.sum(x_s) - np.sum(y_s)

        XY = np.concatenate((self.X, self.Y))
        set_size = int(len(XY) / 2)

        n = 10000
        count = 0

        for i in range(n):
            np.random.shuffle(XY)
            X = XY[0:set_size]
            Y = XY[set_size:]

            x_s = np.array([self.s(x) for x in X])
            y_s = np.array([self.s(y) for y in Y])
            sample_s = np.sum(x_s) - np.sum(y_s)

            import pdb
            pdb.set_trace()

            if sample_s > observed_s:
                count = count + 1

        return count / n
