import json
import numpy as np
import os


# Load and prepare word embeddings
def get_embeddings():
    embeddings = {}
    f = open('glove.840B.300d.txt', encoding='utf8')
    for line in f:
        try:
            line = line.split()
            word = line[0]
            embeddings[word] = line[1:]
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
        words = data[label]["values"]
        embedded_data[label] = []
        for i in words:
            embedded_data[label].append(_get_glove_embedding(i, embeddings_dict))

    with open(embedded_data_file_name, 'w') as results:
        json.dump(embedded_data, results)


# Get embedding of word
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

    for label in data:
        data[label] = [np.asarray(embedding, type='float32') for embedding in data[label]]

    return data


# Calculate effect size
def effect_size(X, Y, A, B):
    x_s = np.array([_s(x, A, B) for x in X])
    y_s = np.array([_s(y, A, B) for y in Y])

    return (np.mean(x_s) - np.mean(y_s)) / np.std(np.concatenate((x_s, y_s)))


# Calculate s(w, A, B)
def _s(w, A, B):
    a_cos = np.array([_cos(w, a) for a in A])
    b_cos = np.array([_cos(w, b) for b in B])

    return (np.mean(a_cos) - np.mean(b_cos))


# Calculate cos(x, y)
def _cos(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


if __name__ == '__main__':
    file_name = 'flowers_insects_pleasant_unpleasant.json'
    embedded_data_file_name = 'flowers_insects_pleasant_unpleasant_embedded.json'
    if not os.path.isfile(embedded_data_file_name):
        glove_embeddings = get_embeddings()

        data = load_targets_and_attributes(file_name)
        embed_data(data, glove_embeddings, embedded_data_file_name)

    embedded_data = load_embedded_data(embedded_data_file_name)
    effect_size = effect_size(embedded_data['target_1'], embedded_data['target_2'], embedded_data['attribute_1'], embedded_data['attribute_2'])
    # p_value = effect_size(embeddings)

    print('file: ' + file_name)
    print('effect_size: ' + str(effect_size))
    # print("p_value:" + p_value)
