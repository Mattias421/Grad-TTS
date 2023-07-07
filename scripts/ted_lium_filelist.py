import os
import re
from tqdm import tqdm



def create_file_list(sph_directory, stm_directory, output_file):

    speaker_dict_file = os.path.join('/store/store4/data/TEDLIUM_release-3/data/wav/speaker_dict.txt')

    with open(output_file, 'w') as file:
        # Process each STM file
        for stm_file in tqdm(os.listdir(stm_directory)):
            stm_path = os.path.join(stm_directory, stm_file)
            with open(stm_path, 'r') as stm_file:
                lines = stm_file.readlines()

                # Process each line in the STM file
                for line in lines:
                    line_parts = line.strip().split()

                    # Extract relevant information from the STM line
                    speaker_name = line_parts[0].split('_')[0]
                    speaker_id = get_speaker_id(speaker_name, speaker_dict_file)
                    start_time = float(line_parts[3])
                    end_time = float(line_parts[4])
                    transcription = ' '.join(line_parts[6:])

                    # Create the corresponding WAV file path
                    wav_file = f"{speaker_id}_{start_time}-{end_time}.wav"
                    wav_file = re.sub(r'(\d+)\.(\d+)', r'\1_\2', wav_file)
                    wav_path = os.path.join(sph_directory, wav_file)

                    # Create the line in the desired format
                    line = f"{wav_path}|{transcription}|{speaker_id}"

                    # Write the line to the output file
                    file.write(line + '\n')

def get_speaker_id(speaker_name, speaker_dict_file):
    with open(speaker_dict_file, 'r') as file:
        for line in file:
            speaker_id, name = line.strip().split(':')
            if name.strip() == speaker_name:
                return speaker_id.strip()

    return None

# Example usage
sph_directory = '/store/store4/data/TEDLIUM_release-3/data/wav'
stm_directory = '/store/store4/data/TEDLIUM_release-3/data/stm'
output_file = 'file_list.txt'

# Create the file list
create_file_list(sph_directory, stm_directory, output_file)
