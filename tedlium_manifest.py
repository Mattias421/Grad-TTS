import os
import re
from sphfile import SPHFile

def convert_utterances(sph_directory, stm_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize speaker ID counter
    speaker_id_counter = 0

    # Initialize dictionary for mapping speaker IDs to names
    speaker_dict = {}

    # Process each STM file
    for stm_file in os.listdir(stm_directory):
        stm_path = os.path.join(stm_directory, stm_file)
        with open(stm_path, 'r') as file:
            lines = file.readlines()

            # Process each line in the STM file
            for line in lines:
                line_parts = line.strip().split()

                # Extract relevant information from the STM line
                speaker_name = line_parts[0].split('_')[0]
                start_time = float(line_parts[3])
                end_time = float(line_parts[4])
                transcription = ' '.join(line_parts[6:])

                # Assign a numerical ID to the speaker if not already assigned
                if speaker_name not in speaker_dict:
                    speaker_id_counter += 1
                    speaker_dict[speaker_name] = speaker_id_counter

                # Get the numerical ID of the speaker
                speaker_id = speaker_dict[speaker_name]

                # Create the corresponding WAV file path
                wav_file = f"{speaker_id}_{start_time}-{end_time}.wav"
                wav_path = os.path.join(output_directory, wav_file)

                # Load the original SPH file and segment it
                sph_file = f"{line_parts[0]}.sph"
                sph_path = os.path.join(sph_directory, sph_file)
                sph = SPHFile(sph_path)
                sph.write_wav(wav_path, start_time, end_time)

    # Save the speaker dictionary as a file
    speaker_dict_file = os.path.join(output_directory, 'speaker_dict.txt')
    with open(speaker_dict_file, 'w') as file:
        for speaker_name, speaker_id in speaker_dict.items():
            file.write(f"{speaker_id} : {speaker_name}\n")

# Example usage
sph_directory = '/store/store4/data/TEDLIUM_release-3/data/sph'
stm_directory = '/store/store4/data/TEDLIUM_release-3/data/stm'
output_directory = '/store/store4/data/TEDLIUM_release-3/data/wav'

# Convert the utterances to WAV files and save the speaker dictionary
convert_utterances(sph_directory, stm_directory, output_directory)
