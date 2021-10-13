import json
import numpy as np

def load_data(file_name):
    f = open(file_name)
    data = json.open(f)
    return data

def embed_data(data):
    embeddings = {}
    for k, v in data:
        words = v["values"]
        embeddings[k] = []
        for i in words:
            embeddings[k].append(glove_embedding(i))
    return embeddings


def effect_size(X, Y, A, B):
    x_s = []
    for x in X:
        x_s.append(_s(x, A, B))
    
    y_s = []
    for y in Y:
        y_s.append(_s(y, A, B))
    
    return (np.mean(x_s) - np.mean(y_s))/np.std(x_s + y_s)


def _s(w, A, B):
    a_cos = []
    for a in A:
        a_cos.append(_cos(w, a))

    b_cos = []
    for b in B:
        b_cos.append(_cos(w, b))
    
    return np.mean(a_cos) - np.mean(b_cos)


def _cos(x, y):
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

def p_value():


def __init__ main():
    file_name = 'flowers_insects_pleasant_unpleasant.json'
    data = load_data(file_name)
    embeddings = embed_data(data)

    effect_size = effect_size(embeddings)
    p_value = effect_size(embeddings)

    print("file: " + file_name)
    print("effect_size: " + effect_size)
    print("p_value:" + p_value)





