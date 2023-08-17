# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import matplotlib.pyplot as plt
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from torch.utils.data import DataLoader


from text import text_to_sequence, cmudict

from likelihood import likelihood, sde_lib

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    args = parser.parse_args()
    

    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')

    score_model = generator.decoder.estimator

    
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
            

    test_dataset = TextMelSpeakerDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)
    
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    batch_collate = TextMelSpeakerBatchCollate()

    loader = DataLoader(dataset=test_dataset, batch_size=1,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=8, shuffle=False)

    for item in loader:
        speech =  item['y'].cuda()
        text = item['x'].cuda()
        spk = item['spk'].cuda()
        x_lengths = item['x_lengths'].cuda()
        y_lengths = item['y_lengths'].cuda()


        # noisy_text = "The thought of poor dead Annie Coulson flashed into Philip's mind."
        # noisy_text = 'That is the kind of hold that curiosity has on the monkeys.'
        # noisy_text = 'I was at this time knitting a pair of white cotton stockings for my mistress; and had not yet wrought upon a Sabbath day.'
        # noisy_text = ' and had not yet listen to black Sabbath today.I was eating a pair of black cotton stockings stockings;'
        noisy_text = 'How much wood would a wood chuck chuck if a wood chuck would chuck wood?'
        # noisy_text = 'hi'
        text = torch.LongTensor(intersperse(text_to_sequence(noisy_text, dictionary=cmu), len(symbols))).cuda()[None]
        x_lengths = torch.LongTensor([text.shape[-1]]).cuda()

        # text = torch.randn_like(text.to(torch.float32)).to(torch.long)
        # speech = torch.randn_like(speech)


        score_model, mu, spk_emb, mask = generator.get_score_model(text.cuda(), x_lengths.cuda(), speech.cuda(), y_lengths.cuda(), spk.cuda())
        sde = sde_lib.SPEECHSDE(beta_min=beta_min, beta_max=beta_max, N=pe_scale, mu=mu, spk=spk_emb, mask=mask)  

        # n_timesteps = 10
        # h = 1.0 / n_timesteps
        # z = mu + torch.randn_like(mu)
        # xt = z * mask
        # for i in range(n_timesteps):
        #     t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
        #                                          device=z.device)
        #     time = t.unsqueeze(-1).unsqueeze(-1)
            
        #     noise_t = beta_min + (beta_max - beta_min) * time
            
        #     dxt = 0.5 * (mu - xt - score_model(xt, t))
        #     dxt = dxt * noise_t * h
        #     xt = (xt - dxt) * mask
        #     import matplotlib.pyplot as plt
        #     plt.imshow(xt.detach().cpu().squeeze())
        #     plt.title(str(t))
        #     plt.savefig(f'../delta_plots/libritts_tts_sampler/{i}.png')

        
        # # generator.decoder.reverse_diffusion(z.cuda(), mask, mu, n_timesteps, stoc=False, spk=generator.spk_emb(spk.cuda()))
        # generator(text, x_lengths, n_timesteps, spk=spk)

        likelihood_fn = likelihood.get_likelihood_fn(sde, lambda x : x)


        score_model = score_model.cuda()

        print('Calculating likelihood')

        print('\n', likelihood_fn(score_model, speech), 'NLL')
        print('Thats a nice likelihood!')



print('Done.')
