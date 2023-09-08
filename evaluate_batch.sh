#!/bin/bash

# Array of values to iterate through
values=(1 25 50 100 150 200 250 300)

# Loop through the values
for n in "${values[@]}"; do
    # Construct the checkpoint file name
    checkpoint_file="/exp/exp4/acq22mc/diff_list/logs/tedlium-1/spk_id/grad_$n.pt"
    
    # Execute the Python script with the appropriate arguments
    python evaluate_tts.py -c ${checkpoint_file} -g 2 
done
