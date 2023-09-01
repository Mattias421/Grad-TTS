# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'resources/filelists/tedlium/train.txt' # tedlium corpus doesn't use filelist
valid_filelist_path = 'resources/filelists/tedlium/dev.txt'
test_filelist_path = 'resources/filelists/tedlium/test.txt'
train_spk = '/store/store4/data/TEDLIUM_release1/tedlium/spk_emb/train.pt'
valid_spk= '/store/store4/data/TEDLIUM_release1/tedlium/spk_emb/dev.pt'
test_spk = '/store/store4/data/TEDLIUM_release1/tedlium/spk_emb/test.pt'
cmudict_path = 'resources/cmu_dictionary'
add_blank = True
n_feats = 80
n_spks = -1  # cheat code to unlock pre-trained speaker embeddings
spk_emb_dim = 192 # speaker encoder dim
n_feats = 80
n_fft = 1024
sample_rate = 16000
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = '../logs/tedlium-1/new_test/'
test_size = 1 
n_epochs = 50
batch_size = 16
learning_rate = 1e-4
seed = 1
save_every = 1
out_size = fix_len_compatibility(2*sample_rate//256)
