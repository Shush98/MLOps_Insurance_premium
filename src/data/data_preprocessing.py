import numpy as np
import pandas as pd
import os
from datetime import datetime, date
import re
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def age(born):
    born = str(born)
    born = datetime.strptime(born, "%Y-%m-%d %H:%M:%S.%f").date()
    today = date.today()
    return today.year - born.year - ((today.month,
                                      today.day) < (born.month,
                                                    born.day))
# Define the preprocessing function
def outlier_removal(df):
    try:
        Q1 = df['Premium Amount'].quantile(0.25)
        Q3 = df['Premium Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR

        upper_array = np.where(df['Premium Amount'] >= upper)[0]
        lower_array = np.where(df['Premium Amount'] <= lower)[0]

        df.drop(index=np.concatenate((upper_array, lower_array)),inplace=True).reset_index(drop=True)

        return df
    except Exception as e:
        logger.error(f"Error in outlier remova;: {e}")
        return df

def preprocess_data(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        df = outlier_removal(df)
        logger.debug('Preprocessing completed')
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw_data_new/train.csv')
        test_data = pd.read_csv('./data/raw_data_new/test.csv')
        logger.debug('Data loaded successfully')

        # Preprocess the data
        train_data['Policy Age'] = train_data['Policy Start Date'].apply(age)
        test_data['Policy Age'] = test_data['Policy Start Date'].apply(age)

        train_data = preprocess_data(train_data)

        # Save the processed data
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()