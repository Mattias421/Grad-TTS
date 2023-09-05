import pickle
import pandas as pd
import numpy as np
from jiwer import wer
import argparse

def get_n_best_list(idx, n_best_list, N=10):
    texts = [n_best_list[idx]['beams'][0][i]['text'] for i in range(N)]
    return texts

def calc_wer(idx, n_best_list):

    scoring = lambda x : x['second_pass_score']
    rescoring = lambda x : x['diffusion_score'] + x['second_pass_score']
    oracle = lambda x : wer(x['text'])

    best_list = n_best_list[idx]['beams'][0].values()

    original_list = sorted(best_list, key=scoring)
    new_list = sorted(best_list, key=rescoring)
    oracle_list = sorted(best_list, key=oracle)



    reference = n_best_list[idx]['targets']
    original_wer = [wer(reference, t['text']) for t in original_list]
    new_wer = [wer(reference, t['text']) for t in new_list]
    oracle_wer = [wer(reference, t['text']) for t in oracle_list]

    print(original_wer[0])
    print(new_wer[0])
    print(oracle_wer[0])

    # reduction = wer(reference, texts[0]) - wer(reference, rescored_texts[0])

    # print(reference)
    # print(texts[0])
    # print(rescored_texts[0])

    return original_wer[0] - new_wer[0]



def main(args):
    with open('/store/store4/data/nbests/tedlium/dev_tmp_out.pkl', 'rb') as f:
        n_best_list = pickle.load(f) 

    N = 5

    results = np.genfromtxt(args.file, delimiter=',').reshape((len(n_best_list), N))

    print(results.shape)
    # print([n_best_list[100]['beams'][0][i]['second_pass_score'] for i in range(N)])

    n_samples = len(n_best_list)

    base_score = lambda x : x['second_pass_score']

    for i in range(n_samples):
        for n in range(N):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = results[i, n]

        for n in range(N, 1000):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = 0

    sum = 0
    n_samples = 30


    for i in range(n_samples):
        sum += calc_wer(i, n_best_list)
    print(sum)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='Results file')
    args = parser.parse_args()
    main(args)


