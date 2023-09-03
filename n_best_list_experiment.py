import pickle
from likelihood import likelihood, sde_lib
from data import TextMelZeroSpeakerDataset, TextMelZeroSpeakerBatchCollate, TextMelSpeakerDataset, TextMelSpeakerBatchCollate

from model import GradTTS
from text.symbols import symbols
from utils import intersperse
from text import text_to_sequence, cmudict

cmu = cmudict.CMUDict('./resources/cmu_dictionary')

import torch
from torch.utils.data import DataLoader
import argparse

import numpy as np
from tqdm import tqdm

from likelihood import likelihood, sde_lib

speaker_id = True

if speaker_id:
    import params_tedlium_spk as params
else:
    import params_tedlium as params
    valid_spk = params.valid_spk
    test_spk = params.test_spk

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
test_filelist_path = params.test_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank


log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale


def get_n_best_list(idx, n_best_list, N=10):
    texts = [n_best_list[idx]['beams'][0][i]['text'] for i in range(N)]
    return texts

def rescore(audio, texts, spk, generator):

    y_lengths = torch.LongTensor([audio.shape[-1]]).cuda()

    def score_text(text):
        text = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
        x_lengths = torch.LongTensor([text.shape[-1]]).cuda()

        score_model, mu, spk_emb, mask = generator.get_score_model(text, x_lengths, audio, y_lengths, spk)
        sde = sde_lib.SPEECHSDE(beta_min=beta_min, beta_max=beta_max, N=pe_scale, mu=mu, spk=spk_emb, mask=mask)  

        likelihood_fn = likelihood.get_likelihood_fn(sde, lambda x : x)

        score_model = score_model.cuda()

        score = likelihood_fn(score_model, audio)

        return score

    new_scores = list(map(score_text, texts))

    return new_scores

def main(args):
    if speaker_id:
        test_dataset = TextMelSpeakerDataset(valid_filelist_path, cmudict_path, add_blank,
                                        n_fft, n_feats, sample_rate, hop_length,
                                        win_length, f_min, f_max)
        batch_collate = TextMelSpeakerBatchCollate()
    else:
        test_dataset = TextMelZeroSpeakerDataset(valid_filelist_path, params.valid_spk, cmudict_path, add_blank,
                                        n_fft, n_feats, sample_rate, hop_length,
                                        win_length, f_min, f_max)
        batch_collate = TextMelZeroSpeakerBatchCollate()
    loader = DataLoader(dataset=test_dataset, batch_size=1,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=False)


    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                            params.n_enc_channels, params.filter_channels,
                            params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                            params.enc_kernel, params.enc_dropout, params.window_size,
                            params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()

    with open('/store/store4/data/nbests/tedlium/dev_tmp_out.pkl', 'rb') as f:
        n_best_list = pickle.load(f) 

    N = 10

    print(f'Rescoring {N} best of {len(n_best_list)} utterances')

    new_scores_dataset = np.zeros((len(n_best_list), N))

    for i, item in tqdm(enumerate(loader)):
        audio = item['y'].cuda()
        spk = item['spk'].cuda()

        texts = get_n_best_list(i, n_best_list, N)

        new_am_scores = rescore(audio, texts, spk, generator)

        new_scores_dataset[i, :] = new_am_scores

        new_scores_dataset.tofile(f'../logs/nbest_exp/{args.name}.csv', sep=',')

        torch.cuda.empty_cache()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-n', '--name', type=str, default='result')
    # parser.add_argument('-s', '--speaker_id', type=bool, choices=[True, False], help='Choose whether to use speaker id or speaker embedding')
    args = parser.parse_args()

    main(args)
    
