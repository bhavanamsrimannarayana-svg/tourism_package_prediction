# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

def clean_and_standardize_pipeline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    A reusable MLOps preprocessing step that cleans and standardizes
    categorical columns using a predefined configuration mapping.
    """
    df = df.copy()
    
    for column, mapping in config.items():
        if column not in df.columns:
            continue
            
        # 1. Canonicalization: strip extra whitespace and lowercase everything
        # This converts 'Fe Male' -> 'female' and 'Self Enquiry' -> 'selfenquiry'
        cleaned_series = (
            df[column]
            .astype(str)
            .str.lower()
            .str.replace(r'\s+', '', regex=True)
        )
        
        # 2. Map variations to canonical forms
        df[column] = cleaned_series.map(mapping)
        
        # 3. Handle unexpected categories gracefully by marking them as 'Unknown'
        df[column] = df[column].fillna('Unknown')
        
    return df

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/srbhavan/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH, index_col=0)
print("Dataset loaded successfully.")

# --- Production Pipeline Configuration ---
# Map the 'space-stripped lowercase' variations to your desired output formats
CLEANING_CONFIG = {
    'Gender': {
        'female': 'Female', 
        'female': 'Female',
        'male': 'Male', 
        'm': 'Male',
        'female': 'Female',
        'female': 'Female'
    },
    'MaritalStatus': {
        'married': 'Married',
        'divorced': 'Divorced',
        'single': 'Single',
        'unmarried': 'Single'  # Could merge with 'Single' if desired
    },
    'TypeofContact': {
        'selfenquiry': 'Self Enquiry',
        'companyinvited': 'Company Invited'
    }
}

# Add 'female' mapping explicitly for the cleaned internal string 'female'
CLEANING_CONFIG['Gender']['female'] = 'Female'

# --- Run the pipeline step on your data ---

print("Before cleaning:")
print(df['Gender'].value_counts())
print(df['MaritalStatus'].value_counts())



# Apply the automated cleaning script
cleaned_df = clean_and_standardize_pipeline(df, CLEANING_CONFIG)

print("\nAfter cleaning:")
print(cleaned_df['Gender'].value_counts())
print(cleaned_df['MaritalStatus'].value_counts())

# Drop the unique identifier
cleaned_df.drop(columns=['CustomerID'], inplace=True)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = cleaned_df.drop(columns=[target_col])
y = cleaned_df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="srbhavan/tourism-package-prediction",
        repo_type="dataset",
    )
