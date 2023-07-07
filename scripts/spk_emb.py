import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
from lhotse import CutSet
from typing import Dict
import re

# Load the pre-trained classifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def transform_txt(txt:str) -> str:
    lower_case, remove_square_brackets, remove_parentheses, remove_triangle_brackets = str.lower, re.compile(r'\[.*?\]'), re.compile(r'\(.*?\)'), re.compile(r'<.*?>')
    remove_curly_brackets, trim, remove_double_spaces, = re.compile(r'\{.*?\}'), str.strip, re.compile(r' +')
    transforms = [
        lower_case,
        lambda x: re.sub(remove_square_brackets, '', x),
        lambda x: re.sub(remove_parentheses, '', x),
        lambda x: re.sub(remove_triangle_brackets, '', x),
        lambda x: re.sub(remove_curly_brackets, '', x),
        trim,
        lambda x: re.sub(remove_double_spaces, ' ', x),
        lambda x: re.sub(" '", "'", x), # that 's -> that's
    ]
    for cmd in transforms:
        txt = cmd(txt)
    return txt


def load_corpus(
        target_folder:str='/store/store4/data/TEDLIUM_release1/tedlium/', 
        prefix_path='/store/store4/data/',
        file_name='tedlium', 
        transform:bool=True
    ) -> Dict[str, CutSet]:
    ds = {}
    for split in ['train', 'dev', 'test']:
        cuts = CutSet.from_file(os.path.join(target_folder, f'{file_name}_cuts_{split}.jsonl.gz'))
        ds[split] = cuts.with_recording_path_prefix(prefix_path)
        if transform: 
            ds[split] = ds[split].transform_text(transform_txt)
    return ds 

for split in ['train', 'dev', 'test']:
    corpus = load_corpus()[split]
    spk_emb_list = []

    for item in tqdm(corpus):
        audio = torch.FloatTensor(item.load_audio())
        spk_emb = classifier.encode_batch(audio).squeeze()
        spk_emb_list.append(spk_emb)

    spk_emb_tensor = torch.stack(spk_emb_list)
    torch.save(spk_emb_tensor, f'/store/store4/data/TEDLIUM_release1/tedlium/spk_emb/{split}.pt')