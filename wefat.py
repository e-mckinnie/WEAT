import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

from wordembeddingtest import WordEmbeddingTest


class WEFAT(WordEmbeddingTest):
    def __init__(self, W, A, B):
        self.W = W
        self.A = A
        self.B = B

    # Return all calculated effect sizes
    def all_effect_sizes(self):
        effect_sizes = {}
        for w in self.W:
            effect_size = self.effect_size(self.W[w])
            effect_sizes[w] = effect_size
        return effect_sizes

    # Calculate s(w, A, B)
    def effect_size(self, w):
        a_cos = np.array([self._cos(w, a) for a in self.A])
        b_cos = np.array([self._cos(w, b) for b in self.B])

        return (np.mean(a_cos) - np.mean(b_cos)) / np.std(np.concatenate((a_cos, b_cos)))

    # Return all calculated p_values
    def all_p_values(self, iterations, distribution_type):
        p_values = {}
        for w in self.W:
            p_value = self.p_value(self.W[w], iterations, distribution_type)
            p_values[w] = p_value
        return p_values

    # Get p-value
    # {(A_i, B_i)}_i is all partitions of A union B into two sets of equal size
    # p-value = Pr_i[s(w, A_i, B_i) > s(w, A, B)]
    def p_value(self, w, iterations, distribution_type):
        test_statistic = self._get_test_statistic(w)
        null_distribution = self._null_distribution(w, iterations)

        if distribution_type == 'normal':
            mu, std = norm.fit(null_distribution)
            cdf = norm.cdf(test_statistic, mu, std)
            return 1 - cdf

        elif distribution_type == 'empirical':
            ecdf = ECDF(null_distribution)
            return 1 - ecdf(test_statistic)

    # Get test statistic
    def _get_test_statistic(self, w):
        a_cos = np.array([self._cos(w, a) for a in self.A])
        b_cos = np.array([self._cos(w, b) for b in self.B])

        return np.mean(a_cos) - np.mean(b_cos)

    # Get null distribution
    def _null_distribution(self, w, iterations):
        AB = np.concatenate((self.A, self.B))
        set_size = int(len(AB) / 2)
        distribution = np.zeros(iterations)

        for i in range(iterations):
            np.random.shuffle(AB)
            A = AB[0:set_size]
            B = AB[set_size:]

            a_cos = np.array([self._cos(w, a) for a in A])
            b_cos = np.array([self._cos(w, b) for b in B])

            distribution[i] = np.mean(a_cos) - np.mean(b_cos)

        return distribution

    # Get name of test
    def tostr(self):
        return 'WEFAT'
