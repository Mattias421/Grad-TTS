import pickle
import pandas as pd
import numpy as np
from jiwer import wer, process_words
import argparse
import pandas as pd
from scipy.optimize import minimize
import os

def get_n_best_list(idx, n_best_list, N=10):
    texts = [n_best_list[idx]['beams'][0][i]['text'] for i in range(N)]
    return texts


def get_best_transcription(idx, n_best_list, alpha):

    reference = n_best_list[idx]['targets']

    # 'first_pass_score', 'am_score', 'bpe_lm_score', 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 'ngram_lm_score_oov', 'ngram_lm_score', 'second_pass_score', 'diffusion_score'
    rescoring = lambda x : np.dot(alpha, list(x.values())[1:])
    # rescoring = lambda x : wer(reference, x['text'])

    best_list = list(n_best_list[idx]['beams'][0].values())

    best_list = best_list[:10]

    new_list = sorted(best_list, key=rescoring)

    return reference[0], new_list[0]['text']



def calc_wer(alpha, n_best_list, n_samples):

    references = []
    news = []

    # 'first_pass_score', 'am_score', 'bpe_lm_score', 
    # 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 'ngram_lm_score_oov', 
    # 'ngram_lm_score', 'second_pass_score', 'diffusion_score'

    for i in range(n_samples):
        ref, new = get_best_transcription(i, n_best_list, alpha)
        references.append(ref)
        news.append(new)

    # print(f'New {wer(references, news)}')


    return wer(references, news)

def main(args):
    with open('/store/store4/data/nbests/tedlium/dev_tmp_out.pkl', 'rb') as f:
        n_best_list = pickle.load(f) 

    N = 10

    results = np.genfromtxt(args.file, delimiter=',').reshape((len(n_best_list), N))

    print(results.shape)

    n_samples = len(n_best_list)

    for i in range(n_samples):
        for n in range(N):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = results[i, n]

        for n in range(N, 1000):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = 0

    # alpha = [0,0,0,0,0,0,0,0,1]
    # print(f'Diffusion only {calc_wer(alpha, n_best_list, n_samples)}')

    wer_table = []

    for i in range(1000):
        alpha = [
            np.random.normal(-2, 0.5),
            np.random.normal(-0, 0),
            np.random.normal(2, 0.5),
            np.random.normal(-2, 0.5),
            0,
            0,
            np.random.normal(-2, 0.5),
            0,
            np.random.normal(0.001, 0.001)
        ]

        w = calc_wer(alpha, n_best_list, n_samples)
    
        wer_table.append(np.append(alpha, w))

    # alpha = [-0.3, -2.4, 0.5, 0.3, 0.3, 0.9, -1.4, -0.3, 0.02]

    # result = minimize(calc_wer, alpha, (n_best_list, n_samples), method='Nelder-mead')

    # print(result.x)

    df = pd.DataFrame(wer_table, columns=['first_pass_score', 'am_score', 'bpe_lm_score', 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 
                       'ngram_lm_score_oov', 'ngram_lm_score', 'second_pass_score', 'diffusion_score', 'wer'])
    print(df[df.wer == df.wer.min()])

    directory, filename = os.path.split(args.file)
    filename_without_extension, file_extension = os.path.splitext(filename)

    new_filename = f"{filename_without_extension}_wer{file_extension}"
    new_file_path = os.path.join(directory, new_filename)

    df.to_csv(new_file_path)

if __name__ == '__main__':
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='Results file')
    args = parser.parse_args()

    main(args)


