import pickle
import pandas as pd
import numpy as np
from jiwer import wer, process_words
import pandas as pd
from scipy.optimize import minimize
import os

import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import yaml

log = logging.getLogger(__name__)

def get_n_best_list(idx, n_best_list, N=10):
    texts = [n_best_list[idx]['beams'][0][i]['text'] for i in range(N)]
    return texts


def get_best_transcription(idx, n_best_list, alpha, N):

    reference = n_best_list[idx]['targets']

    # 'first_pass_score', 'am_score', 'bpe_lm_score', 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 'ngram_lm_score_oov', 'ngram_lm_score', 'second_pass_score', 'diffusion_score'
    linear = lambda x : np.dot(alpha, list(x.values())[1:])
    log_linear = lambda x : np.exp(linear(x))
    # rescoring = lambda x : wer(reference, x['text'])

    best_list = list(n_best_list[idx]['beams'][0].values())

    best_list = best_list[:N]

    new_list = sorted(best_list, key=linear)

    am_scores = [x['am_score'] for x in new_list]
    diff_scores = [x['diffusion_score'] for x in new_list]

    # log.info(f'Mean AM score is {np.mean(am_scores)}, std is {np.std(am_scores)}') 
    # log.info(f'Mean diff score is {np.mean(diff_scores)}, std is {np.std(diff_scores)}') 

    return reference[0], new_list[0]['text']



def calc_wer(alpha, n_best_list, n_samples, N):

    references = []
    news = []

    # 'first_pass_score', 'am_score', 'bpe_lm_score', 
    # 'first_pass_length_penalty', 'ngram_lm_score_non_oov', 'ngram_lm_score_oov', 
    # 'ngram_lm_score', 'second_pass_score', 'diffusion_score'

    for i in range(n_samples):
        ref, new = get_best_transcription(i, n_best_list, alpha, N)
        references.append(ref)
        news.append(new)

    # print(f'New {wer(references, news)}')


    return wer(references, news)

@hydra.main(version_base=None, config_path='./config', config_name='rescoring')
def main(cfg):
    np.random.seed(cfg.seed)
    with open(cfg.n_best_list, 'rb') as f:
        n_best_list = pickle.load(f) 

    N = cfg.n

    diff_scores = np.load(cfg.diff_score_list).reshape((len(n_best_list), N))

    print(diff_scores[0])

    len_scores = np.zeros(diff_scores.shape)

    for i in range(len_scores.shape[0]):
        for n in range(len_scores.shape[1]):
            len_scores[i, n] = n_best_list[i]['beams'][0][n]['first_pass_length_penalty']

    am_scores = np.zeros(diff_scores.shape)

    for i in range(am_scores.shape[0]):
        for n in range(am_scores.shape[1]):
            am_scores[i, n] = n_best_list[i]['beams'][0][n]['am_score']

    sp_scores = np.zeros(diff_scores.shape)

    for i in range(sp_scores.shape[0]):
        for n in range(sp_scores.shape[1]):
            sp_scores[i, n] = n_best_list[i]['beams'][0][n]['second_pass_score']

    fp_scores = np.zeros(diff_scores.shape)

    for i in range(fp_scores.shape[0]):
        for n in range(fp_scores.shape[1]):
            fp_scores[i, n] = n_best_list[i]['beams'][0][n]['first_pass_score']



    log.info(diff_scores.shape)

    log.info(f'Total diff mean {np.mean(diff_scores)}, std {np.std(diff_scores)}') 
    log.info(f'Total am mean {np.mean(am_scores)}, std {np.std(am_scores)}') 
    log.info(f'beam diff mean {np.mean(diff_scores, axis=0)}, std {np.std(diff_scores, axis=0)}') 
    log.info(f'am mean {np.mean(am_scores,axis=0)}, std {np.std(am_scores,axis=0)}') 


    plt.plot(np.std(am_scores, axis=0))
    plt.savefig('plots/am_std')
    plt.clf()
    plt.plot(np.std(diff_scores, axis=0))
    plt.savefig('plots/diff_std')
    plt.clf()
    plt.plot(np.mean(am_scores, axis=0))
    plt.savefig('plots/am_mean')
    plt.clf()
    plt.plot(np.mean(diff_scores, axis=0))
    plt.savefig('plots/diff_mean')
    plt.clf()
    plt.plot(np.mean(len_scores, axis=0))
    plt.savefig('plots/len_mean')
    plt.clf()
    plt.plot(np.mean(sp_scores, axis=0))
    plt.savefig('plots/sp_mean')
    plt.clf()
    plt.plot(np.mean(fp_scores, axis=0))
    plt.savefig('plots/fp_mean')

    n_samples = len(n_best_list)

    for i in range(n_samples):
        for n in range(N):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = diff_scores[i, n]

        for n in range(N, 1000):
            n_best_list[i]['beams'][0][n]['diffusion_score'] = 0

    alpha = list(cfg.weights.values())
    result = calc_wer(alpha, n_best_list, n_samples, N)

    log.info(result)

    out = dict(cfg.weights)
    out['wer'] = result
    out['diff_config'] = cfg.diff_score_list.split('/')[-1].split('.')[0]

    with open('result.yaml', 'w') as yaml_file:
        yaml.dump(out, yaml_file)

    return result
        
if __name__ == '__main__':
    main()


