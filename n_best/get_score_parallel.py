import pickle
from likelihood import likelihood, sde_lib

import sys
sys.path.append('../')
from data import TextMelSpeakerDataset, TextMelSpeakerBatchCollate
from model.utils import fix_len_compatibility

from model import GradTTS
from text.symbols import symbols
from text import cmudict

cmu = cmudict.CMUDict('/mnt/parscratch/users/acq22mc/exp/Grad-TTS/resources/cmu_dictionary')

import torch

import numpy as np
from tqdm import tqdm
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
log = logging.getLogger()


import params_tedlium_spk as params


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
test_filelist_path = params.test_filelist_path
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


def rescore(batch, generator, device, n_euler):

    text = batch['x'].to(device)[None]
    audio = batch['y'].to(device)[None]
    spk = batch['spk'].to(device)
    x_lengths = torch.LongTensor([text.shape[-1]]).to(device)
    y_lengths = torch.LongTensor([audio.shape[-1]]).to(device)

    y_max_length = fix_len_compatibility(y_lengths[0])
    y = torch.zeros((1, n_feats, y_max_length), dtype=torch.float32)
    y[0, :, :audio.shape[-1]] = audio
    audio = y

    score_model, mu, spk_emb, mask = generator.get_score_model(text, x_lengths, audio, y_lengths, spk)
    sde = sde_lib.SPEECHSDE(beta_min=beta_min, beta_max=beta_max, N=pe_scale, mu=mu, spk=spk_emb, mask=mask)  

    likelihood_fn = likelihood.get_likelihood_fn(sde, lambda x : x, rtol=1e-3, atol=1e-3, euler=n_euler)

    score_model = score_model.to(device)

    score = likelihood_fn(score_model, audio)

    return score.cpu().numpy()[0]


class NBestDataset(torch.utils.data.Dataset):
    def __init__(self, text_mel_dataset, n_best_list, N):
        self.text_meldataset = text_mel_dataset
        self.n_best_list = n_best_list
        self.N = N

    def __len__(self):
        return len(self.text_meldataset) * self.N
    
    def get_n_best_hypothesis(self, item_idx, n_idx):
        text = self.n_best_list[item_idx]['beams'][0][n_idx]['text']
        
        if len(text.strip(' ')) == 0:
            text += ' '
        return self.text_meldataset.get_text(text)
    
    def __getitem__(self, idx):
        item_idx = int(np.floor(idx / self.N))
        n_idx = idx % self.N

        item = self.text_meldataset[item_idx]

        item['x'] = self.get_n_best_hypothesis(item_idx, n_idx)
        return item
    

@hydra.main(version_base=None, config_path='./config')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')

    test_dataset = TextMelSpeakerDataset(cfg.filelist, '/mnt/parscratch/users/acq22mc/exp/Grad-TTS/resources/cmu_dictionary', add_blank,
                                    n_fft, n_feats, sample_rate, hop_length,
                                    win_length, f_min, f_max)

    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                            params.n_enc_channels, params.filter_channels,
                            params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                            params.enc_kernel, params.enc_dropout, params.window_size,
                            params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    generator.load_state_dict(torch.load(cfg.checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.to(device).eval()

    with open(cfg.n_best_list, 'rb') as f:
        n_best_list = pickle.load(f) 

    N = cfg.N

    dataset = NBestDataset(test_dataset, n_best_list, N)
    idx = int(cfg.n_best_dataset_index)
    item_idx = int(np.floor(idx / N))
    n_idx = idx % N

    batch = dataset[idx]

    score = rescore(batch, generator, device, cfg.n_euler)

    log.info(score)

    output = {'i':item_idx, 'n':n_idx, 'N':N, 'name':cfg.name, 'diffusion_score':float(score)}

    with open(f'{item_idx}_{n_idx}.yaml', 'w') as yaml_file:
        yaml.dump(output, yaml_file)
       

if __name__ == '__main__':
    main()
    
