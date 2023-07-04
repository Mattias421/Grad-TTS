import os
import re

folder_path = "/store/store4/data/TEDLIUM_release-3/data/wav/"

# Iterate over all files in the folder
for file_name in os.listdir(folder_path):
    # Construct the old file path
    old_file_path = os.path.join(folder_path, file_name)
    
    # Extract the file extension
    file_extension = os.path.splitext(file_name)[1]
    
    # Create the new file name by replacing dots between numbers with underscores
    new_file_name = re.sub(r'(\d+)\.(\d+)', r'\1_\2', file_name)
    
    # Construct the new file path
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # Rename the file
    os.rename(old_file_path, new_file_path)