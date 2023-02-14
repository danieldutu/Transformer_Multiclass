from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class TextLabelDataset(Dataset):
    """
    Dataset for text and labels

    Args:
        dataframe (pd.DataFrame): Dataframe containing the text and labels
        max_length (int): Maximum length of the text

    Returns:
        torch.Tensor: Input ids
        torch.Tensor: Attention mask
        torch.Tensor: Labels
    """
    def __init__(self, dataframe, max_length):
        self.dataframe = dataframe
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row['text']
        labels = row[['env_problems', 'pollution', 'treatment', 'climate', 'biomonitoring']].values
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors='pt')
        return inputs, labels.astype('float32')


if __name__ == '__main__':

    import pandas as pd
    from torch.utils.data import DataLoader

    df = pd.read_csv(r'D:\NLP\nlp-reports-news-classification\water_problem_nlp_en_for_Kaggle_100.csv', delimiter=';',
                     header=0, encoding='cp1251')

    dataset = TextLabelDataset(df, 512)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        inputs, labels = batch
        print(inputs['input_ids'].shape)
        print(inputs['attention_mask'].shape)
        print(labels.shape)
        break

# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class NLPTaskDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#         self.text = self.data['text'].tolist()
#         self.labels = self.data[self.data.columns[1:]].values.astype(float)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         text = self.data.iloc[idx]['text']
#         label = torch.tensor(self.data.iloc[idx][self.data.columns[1:]]).values
#         sample = {'text': text, 'label': label}
#         return sample
