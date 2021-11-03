import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from weat import WEAT
from wefat import WEFAT


def main(args):
    if not os.path.isfile(args.embedded_data_file_name):
        data = load_targets_and_attributes(args.data_file_name)
        glove_embeddings = get_embeddings(args.glove_file_name)
        embed_data(data, glove_embeddings, args.embedded_data_file_name)

    embedded_data = load_embedded_data(args.embedded_data_file_name)

    if args.test == 'WEAT':
        print(f'Running WEAT on {args.data_file_name}')
        test_weat(embedded_data, args.iterations, args.distribution_type)
    elif args.test == 'WEFAT':
        print(f'Running WEFAT on {args.data_file_name}')
        test_wefat(embedded_data, args.wefat_association_file_name, args.iterations, args.distribution_type)
    else:
        print('Test type not recognized.')


# Load data file with target and attribute words
def load_targets_and_attributes(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    return data


# Load and prepare word embeddings
def get_embeddings(glove_file_name):
    f = open(glove_file_name, encoding='utf8')
    embeddings = {}
    for line in f:
        try:
            line_array = line.split()
            word = line_array[0]
            embeddings[word] = line
        except Exception:
            pass
    return embeddings


# Create file of the embeddings of target and attribute words
def embed_data(data, embeddings_dict, embedded_data_file_name):
    embedded_data = {}
    for label in data:
        words = data[label]
        embedded_data[label] = []
        for i in words:
            embedded_vector = embeddings_dict.get(i)
            if embedded_vector is None:
                print(f'Embedding not found for "{i}"; skipped')
            else:
                embedded_data[label].append(embedded_vector)

    with open(embedded_data_file_name, 'w') as results:
        json.dump(embedded_data, results)


# Load embeddings of target and attribute words
def load_embedded_data(embedded_data_file_name):
    with open(embedded_data_file_name, 'r') as file:
        data = json.load(file)

    embeddings = {}
    for label in data:
        embeddings[label] = {embedding.split()[0]: np.asarray(embedding.split()[1:], dtype='float32') for embedding in data[label]}

    return embeddings


# Run WEAT technique and report effect size and p-value
def test_weat(embedded_data, iterations, distribution_type):
    test = WEAT(list(embedded_data['target_1'].values()), list(embedded_data['target_2'].values()), list(embedded_data['attribute_1'].values()), list(embedded_data['attribute_2'].values()))

    d = test.effect_size()
    print(f'\teffect size: {d}')

    p_value = test.p_value(iterations, distribution_type)
    print(f'\tp_value: {p_value}')


# Run WEFAT technique and show either plot and Pearson correlation coefficient or effect sizes and p-values
def test_wefat(embedded_data, wefat_association_file_name, iterations, distribution_type):
    test = WEFAT(embedded_data['target'], list(embedded_data['attribute_1'].values()), list(embedded_data['attribute_2'].values()))
    s = test.all_effect_sizes()

    if wefat_association_file_name is not None:
        with open(args.wefat_association_file_name, 'r') as file:
            association_data = json.load(file)

            pairs = []
            for association in association_data:
                if association in s.keys():
                    pairs.append((association_data[association], s[association]))

            plt.scatter(*zip(*pairs))
            plt.show()

            correlation_coefficient = np.corrcoef(*zip(*pairs))[0][1]
            print(f'\tPearson\'s correlation coefficient: {correlation_coefficient}')
    else:
        p_values = test.all_p_values(iterations, distribution_type)
        for target_word in s.keys():
            print(f'\t{target_word}')
            print(f'\t\teffect_size: {s[target_word]}')
            print(f'\t\tp_value: {p_values[target_word]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute WEAT or WEFAT.')
    parser.add_argument('--data_file_name', dest='data_file_name', help='file name for target and attribute words. See weat_1.json or wefat_1.json for format')
    parser.add_argument('--embedded_data_file_name', dest='embedded_data_file_name', help='file name for existing or to be created file with embeddings for target and attribute words')
    parser.add_argument('--glove_file_name', dest='glove_file_name', help='file name for GloVE , must be provided if embedded_file_name file does not exist')
    parser.add_argument('--wefat_association_file_name', dest='wefat_association_file_name', help='mapping of target to other statistic, such as occupation to % women. See wefat_1_percentage_women.json for format')
    parser.add_argument('--test', dest='test', help='WEAT or WEFAT')
    parser.add_argument('--iterations', dest='iterations', type=int, help='number of iterations to compute p-value')
    parser.add_argument('--distribution_type', dest='distribution_type', choices=['normal', 'empirical'], help='type of distribution to compute p-value')

    args = parser.parse_args()
    main(args)
