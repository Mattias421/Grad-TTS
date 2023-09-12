import os
import pandas as pd
import yaml

# Define the directory containing the YAML files
directory_path = '/exp/exp4/acq22mc/diff_list/hydra_outputs/n_best_list/'

# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame()

# Loop through files in the directory
for root, _, files in os.walk(directory_path):
    for filename in files:
        if filename.endswith('result.yaml'):
            file_path = os.path.join(root, filename)
            
            # Read the YAML file
            with open(file_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
                
                # Convert the YAML data to a DataFrame (assuming YAML structure is consistent)
                df = pd.DataFrame([yaml_data])
                
                # Append the DataFrame to the result DataFrame
                result_df = result_df._append(df, ignore_index=True)

# Display the final result DataFrame
result_df = result_df.sort_values('wer')
result_df.to_csv('results.csv')
