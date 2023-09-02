import pickle
from likelihood import likelihood, sde_lib
from data import TextMelZeroSpeakerDataset, TextMelZeroSpeakerBatchCollate

import params_tedlium as params
from model import GradTTS
from text.symbols import symbols
from utils import intersperse
from text import text_to_sequence, cmudict

cmu = cmudict.CMUDict('./resources/cmu_dictionary')

import torch
from torch.utils.data import DataLoader
import argparse

from likelihood import likelihood, sde_lib

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
test_filelist_path = params.test_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

valid_spk = params.valid_spk
test_spk = params.test_spk

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

    y_lengths = torch.LongTensor([audio.shape[-1]])

    diffusion_am_scores = []

    for text in texts:
        text = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
        x_lengths = torch.LongTensor([text.shape[-1]])

        score_model, mu, spk_emb, mask = generator.get_score_model(text.cuda(), x_lengths.cuda(), audio.cuda(), y_lengths.cuda(), spk.cuda())
        sde = sde_lib.SPEECHSDE(beta_min=beta_min, beta_max=beta_max, N=pe_scale, mu=mu, spk=spk_emb, mask=mask)  

        likelihood_fn = likelihood.get_likelihood_fn(sde, lambda x : x)


        score_model = score_model.cuda()

        logp0 = likelihood_fn(score_model, audio.cuda())

        diffusion_am_scores.append(logp0)

    return diffusion_am_scores

def main(args):
    test_dataset = TextMelZeroSpeakerDataset(valid_filelist_path, valid_spk, cmudict_path, add_blank,
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

    print(f'Rescoring {len(n_best_list)} utterances')

    for i, item in enumerate(loader):
        audio = item['y']
        spk = item['spk']

        texts = get_n_best_list(i, n_best_list)

        rescore(audio, texts, spk, generator)

        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    args = parser.parse_args()

    main(args)
    
