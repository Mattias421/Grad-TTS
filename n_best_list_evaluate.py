import pickle
import pandas as pd
import numpy as np
from jiwer import wer, process_words
import argparse
import pandas as pd

def get_n_best_list(idx, n_best_list, N=10):
    texts = [n_best_list[idx]['beams'][0][i]['text'] for i in range(N)]
    return texts

def get_best_transcription(idx, n_best_list, alpha):

    reference = n_best_list[idx]['targets']

    # 'first_pass_score', 'am_score', 'bpe_lm_score', 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 'ngram_lm_score_oov', 'ngram_lm_score', 'second_pass_score', 'diffusion_score'
    rescoring = lambda x : np.dot(alpha, list(x.values())[1:])

    best_list = list(n_best_list[idx]['beams'][0].values())

    new_list = sorted(best_list, key=rescoring)

    return reference[0], new_list[0]['text']



def calc_wer(n_best_list, n_samples):

    references = []
    news = []

    # 'first_pass_score', 'am_score', 'bpe_lm_score', 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 'ngram_lm_score_oov', 'ngram_lm_score', 'second_pass_score', 'diffusion_score'
    alpha = np.random.normal(loc=0, scale=1, size=9)
    print(f'alpha = {alpha}')

    for i in range(n_samples):
        ref, new = get_best_transcription(i, n_best_list, alpha)

        references.append(ref)
        news.append(new)

    print(f'New {wer(references, news)}')


    return np.append(alpha, wer(references, news))

def main(args):
    with open('/store/store4/data/nbests/tedlium/dev_tmp_out.pkl', 'rb') as f:
        n_best_list = pickle.load(f) 

    N = 5

    results = np.genfromtxt(args.file, delimiter=',').reshape((len(n_best_list), N))

    print(results.shape)

    n_samples = len(n_best_list)

    for i in range(n_samples):
        for n in range(N):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = results[i, n]

        for n in range(N, 1000):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = 0

    wer_table = []
    
    for i in range(1000):
        wer_table.append(calc_wer(n_best_list, n_samples))

    df = pd.DataFrame(wer_table, columns=['first_pass_score', 'am_score', 'bpe_lm_score', 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 
                       'ngram_lm_score_oov', 'ngram_lm_score', 'second_pass_score', 'diffusion_score', 'wer'])
    print(df[df.wer == df.wer.min()])

    df.to_csv('diffusion_wer.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='Results file')
    args = parser.parse_args()

    main(args)


