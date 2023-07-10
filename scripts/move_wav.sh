#!/bin/bash

# Specify the directory containing the WAV files
directory="/store/store4/data/TEDLIUM_release-3/data/wav"

# Change to the directory
cd "$directory"

# Loop through the WAV files
for file in *.wav; do
  # Extract the number from the beginning of the filename
  number=$(echo "$file" | grep -o "^[0-9]*")

  # Create a new directory based on the number
  mkdir -p "$number"

  # Move the file into the corresponding directory
  mv "$file" "$number/"
done
