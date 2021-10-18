import numpy as np
from sklearn.linear_model import LinearRegression

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

    # Get name of test
    def tostr(self):
        return "WEFAT"

    # Get p-value
    # Uses linear regression to predict p_w from s(w, A, B)
    def p_value(self, pairs):
        np.random.shuffle(pairs)
        new_x = np.array([y for (x, y) in pairs])
        new_y = np.array([x for (x, y) in pairs])

        break_point = int(len(new_y) * 0.9)

        train_x = new_x[0:break_point].reshape((-1, 1))
        test_x = new_x[break_point:].reshape((-1, 1))
        train_y = new_y[0:break_point]
        test_y = new_y[break_point:]

        model = LinearRegression().fit(train_x, train_y)

        count = 0
        n = len(test_y)

        for i in range(n):
            pred_y = model.predict(test_x[i].reshape((-1, 1)))

            if pred_y > test_y[i]:
                count += 1

        return count / n
