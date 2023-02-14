import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import string
import re
from dataprocessor import DataProcessor
from tqdm import tqdm
from dataloader import TextLabelDataset
from model import TransformerModel



# Load data

path = r'D:\NLP\nlp-reports-news-classification\water_problem_nlp_en_for_Kaggle_100.csv'

df = pd.read_csv(path, delimiter=';', header=0, encoding='cp1251')
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data loaders
train_dataset = TextLabelDataset(train_data, max_length=128)
val_dataset = TextLabelDataset(val_data, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Set up the model
model = TransformerModel(output_dim=5, hidden_dim=768, num_layers=4, dropout=0.1)

# Set up the loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    # Train on the training data
    train_loss = 0
    model.train()
    for inputs, labels in train_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        print(f"inputs['input_ids'].shape = {inputs['input_ids'].shape}")
        print(f"inputs['attention_mask'].shape = {inputs['attention_mask'].shape}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Evaluate on the validation data
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    # Print the loss for this epoch
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")


