import pandas as pd
import ast
import pickle
import os

def read_and_fix_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['episode', 'step', 'state', 'action', 'reward', 'done', 'truncated'])
    
    # Convert 'state' column from string representation to actual list or numpy array if needed
    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing {x}: {e}")
            return x  # Or handle the error as appropriate

    #df['state'] = df['state'].apply(lambda x: safe_literal_eval(x) if pd.notnull(x) else x)
    
    # Print the DataFrame to verify
    print(df.head())
    print(df.describe().T)
    
    return df

def save_to_pickle(df, pickle_file_path):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"DataFrame saved to {pickle_file_path}")

def load_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        df = pickle.load(f)
    print("DataFrame loaded successfully")
    print(df.head())
    return df

# Usage

try:
    
    user_input = int(input("Which file would you like to convert from .csv to .pkl (from 1 to 4): "))
    
    current_path = os.getcwd()
    
    csv_file_path = f'{current_path}/PandaPickAndPlaceDense-medium-v3-{user_input}.csv'
    pickle_file_path = f'{current_path}/PandaPickAndPlaceDense-medium-v3-{user_input}.pkl'

# Read and fix CSV
    df = read_and_fix_csv(csv_file_path)

# Save to pickle
    save_to_pickle(df, pickle_file_path)

# Load from pickle
#df_loaded = load_from_pickle(pickle_file_path)

