import pickle
import pandas as pd
import numpy as np
from jiwer import wer

def get_n_best_list(idx, n_best_list, N=10):
    texts = [n_best_list[idx]['beams'][0][i]['text'] for i in range(N)]
    return texts

with open('/store/store4/data/nbests/tedlium/dev_tmp_out.pkl', 'rb') as f:
    n_best_list = pickle.load(f) 

N = 10

results = np.genfromtxt('../logs/nbest_exp/result.csv', delimiter=',').reshape((len(n_best_list), N))

print(results.shape)
# print([n_best_list[100]['beams'][0][i]['second_pass_score'] for i in range(N)])

def calc_wer(idx):
    texts = get_n_best_list(idx, n_best_list)
    scores = results[idx]

    pairs = sorted(list(zip(texts, scores)), key=lambda x : x[1])
    rescored_texts = [pair[0] for pair in pairs]

    reference = n_best_list[idx]['targets']

    reduction = wer(reference, texts[0]) - wer(reference, rescored_texts[0])

    # print(reference)
    # print(texts[0])
    # print(rescored_texts[0])

    # print(f'original wer = {wer(reference, texts[0])}')
    # print(f'new wer = {wer(reference, rescored_texts[0])}')

    return reduction

sum = 0
for i in range(24):
    sum += calc_wer(i)
print(sum)


