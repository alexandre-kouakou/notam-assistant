from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.config import DATASET_NAME, RANDOM_SEED, VAL_SPLIT_FRACTION

def load_train_val_test():
    """
    Load the DEEL-AI/NOTAM dataset from Hugging Face and produce three splits.
    The Hugging Face dataset provides train (4237 rows) and test (4241 rows).
    This function further splits the train set into train and val, stratified by 
    label to preserve class distribution across the split. The test set is
    returned as-is and should not be touched until final evaluation.
    
    Returns
    -------
    (train_df, val_df, test_df) : tuple of pandas.DataFrame
    Each DataFrame has columns 'text' (str) and 'label' (int class id).
    Approximate sizes: train ~3390, val ~847, test 4241.
    """
    ds = load_dataset(DATASET_NAME)

    full_train_df = ds["train"].to_pandas().copy()
    test_df = ds["test"].to_pandas().copy()

    train_df, val_df = train_test_split(full_train_df, test_size = VAL_SPLIT_FRACTION, random_state = RANDOM_SEED, stratify = full_train_df["label"])

    return train_df, val_df, test_df

