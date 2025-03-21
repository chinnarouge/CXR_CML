import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load the dataset
data = pd.read_csv('/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/complete_training_without_lateral_new.csv')

# Separate metadata and labels
metadata_columns = data.columns[:6]  # First 6 columns are metadata
label_columns = data.columns[6:]     # Remaining 40 columns are disease labels

# Function to perform 5-fold cross-validation
def kfold_split(data, labels, n_splits=5):
    """
    Perform K-Fold cross-validation split while maintaining label distribution.

    Args:
        data (DataFrame): Full dataset including metadata and labels.
        labels (DataFrame): Only the label columns.
        n_splits (int): Number of folds.

    Returns:
        None: Saves each fold's train and validation set as CSV files.
    """
    # Combine labels to create stratification keys
    stratify_keys = labels.apply(lambda x: ''.join(x.astype(str)), axis=1)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Generate and save train-validation splits
    for i, (train_indices, valid_indices) in enumerate(skf.split(data, stratify_keys)):
        train_fold = data.iloc[train_indices]
        valid_fold = data.iloc[valid_indices]

        train_fold.to_csv(f'/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/training/train_fold_{i+1}.csv', index=False)
        valid_fold.to_csv(f'/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/valid_fold_{i+1}.csv', index=False)

# Perform 5-fold cross-validation
kfold_split(data, data[label_columns])

print("5-Fold cross-validation splits saved as CSV files.")
