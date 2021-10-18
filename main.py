import argparse
import json
import numpy as np
import os

from weat import WEAT


def main(args):
    if not os.path.isfile(args.embedded_data_file_name):
        data = load_targets_and_attributes(args.data_file_name)
        glove_embeddings = get_embeddings(args.glove_file_name)
        embed_data(data, glove_embeddings, args.embedded_data_file_name)

    embedded_data = load_embedded_data(args.embedded_data_file_name)

    if args.test == 'WEAT':
        test = WEAT(embedded_data['target_1'], embedded_data['target_2'], embedded_data['attribute_1'], embedded_data['attribute_2'])
    elif args.test == 'WEFAT':
        print('Not yet implemented.')
    else:
        print('Test type not recognized.')

    d = test.effect_size()
    # p_value = test.p_value()

    print(test.tostr())
    print(f'\tdata file: {args.data_file_name}')
    print(f'\teffect size: {d}')
    # print(f'p_value: {p_value}')


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


# Load data file with target and attribute words
def load_targets_and_attributes(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    return data


# Create file of the embeddings of target and attribute words
def embed_data(data, embeddings_dict, embedded_data_file_name):
    embedded_data = {}
    for label in data:
        words = data[label]
        embedded_data[label] = []
        for i in words:
            embedded_data[label].append(_get_glove_embedding(i, embeddings_dict))

    with open(embedded_data_file_name, 'w') as results:
        json.dump(embedded_data, results)


# Get embedding of word from dictionary
def _get_glove_embedding(word, embeddings_dict):
    embedded_vector = embeddings_dict.get(word)
    if embedded_vector is None:
        raise Exception("Embedding not found")
    else:
        return embedded_vector


# Load embeddings of target and attribute words
def load_embedded_data(embedded_data_file_name):
    with open(embedded_data_file_name, 'r') as file:
        data = json.load(file)

    embeddings = {}
    for label in data:
        embeddings[label] = [np.asarray(embedding.split()[1:], dtype='float32') for embedding in data[label]]

    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute WEAT or WEFAT.')
    parser.add_argument('--data_file_name', dest='data_file_name', help='file name for target and attribute words. See flowers_insects-pleasant_unpleasant.json for format')
    parser.add_argument('--embedded_data_file_name', dest='embedded_data_file_name', help='file name for existing or to be created file with embeddings for target and attribute words')
    parser.add_argument('--glove_file_name', dest='glove_file_name', help='file name for GloVE , must be provided if embedded_file_name file does not exist')
    parser.add_argument('--test', dest='test', help='WEAT or WEFAT')

    args = parser.parse_args()
    main(args)
