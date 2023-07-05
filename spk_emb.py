import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

# Load the pre-trained classifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Set the input and output folder paths
input_folder = "/store/store4/data/TEDLIUM_release1/tedlium/wavs"
output_folder = "/store/store4/data/TEDLIUM_release1/tedlium/spk_embs"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Recursive function to process subdirectories
def process_directory(input_dir, output_dir):
    for filename in tqdm(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".wav"):
            # Load the WAV file
            waveform, sample_rate = torchaudio.load(file_path)

            # Generate embeddings
            embeddings = classifier.encode_batch(waveform)

            # Save the embeddings as a .pt file
            output_filename = os.path.splitext(filename)[0] + ".pt"
            output_path = os.path.join(output_dir, output_filename)

            # Create the output subdirectory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            torch.save(embeddings, output_path)


        elif os.path.isdir(file_path):
            # Recursively process subdirectories
            new_output_dir = os.path.join(output_dir, filename)
            process_directory(file_path, new_output_dir)

# Start processing from the top-level input directory
process_directory(input_folder, output_folder)

