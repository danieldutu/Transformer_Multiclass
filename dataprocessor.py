import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Class to process the data

    Parameters
    ----------
    path: str

    test_size: float
        The proportion of the dataset to include in the test split

    random_state: int
        Controls the shuffling applied to the data before applying the split

    Attributes
    ----------
    df: pd.DataFrame
        The dataframe containing the data

    train_data: pd.DataFrame
        The training data

    val_data: pd.DataFrame
        The validation data

    Methods
    -------

    read_csv(path)
        Reads the csv file and returns a dataframe

    preprocess_text(text)
        Preprocesses the text

    split_data(test_size, random_state)
        Splits the data into training and validation sets

    """
    def __init__(self, path, test_size=0.2, random_state=42):
        self.df = self.read_csv(path)
        self.train_data, self.val_data = self.split_data(test_size, random_state)

    def read_csv(self, path):
        df = pd.read_csv(path, delimiter=';', header=0, encoding='cp1251')
        df = df.fillna(0)
        return df

    def preprocess_text(self, text):
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        # Remove digits
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_data(self, test_size, random_state):
        return train_test_split(self.df, test_size=test_size, random_state=random_state)





if __name__ == '__main__':
    data_file = r'D:\NLP\nlp-reports-news-classification\water_problem_nlp_en_for_Kaggle_100.csv'
    data_processor = DataProcessor(data_file)
    print(data_processor.train_data.head())
    print(data_processor.val_data.head())