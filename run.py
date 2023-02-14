import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from dataloader import TextLabelDataset
from model import TransformerModel
import logging


logging.basicConfig(level=logging.INFO)
# Load data


path = r'D:\NLP\nlp-reports-news-classification\water_problem_nlp_en_for_Kaggle_100.csv'

df = pd.read_csv(path, delimiter=';', header=0, encoding='cp1251')

if df.isna().any().any():
    print("DataFrame contains NaN values")
else:
    print("DataFrame does not contain NaN values")
print(df.isna().sum())
df = df.fillna(0)


train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data loaders
train_dataset = TextLabelDataset(train_data, max_length=128)
val_dataset = TextLabelDataset(val_data, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Set up the model
model = TransformerModel(output_dim=5, hidden_dim=768, num_layers=4, dropout=0.1)

# Set up the loss and optimizer
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train the model
num_epochs = 3
for epoch in tqdm(range(num_epochs)):
    # Train on the training data
    train_loss = 0
    model.train()
    for inputs, labels in train_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # print(f"inputs['input_ids'] = {inputs['input_ids']}")
        # print(f"inputs['attention_mask'] = {inputs['attention_mask']}")
        labels = labels.to(device)

        # print(f"inputs['input_ids'].shape = {inputs['input_ids'].shape}")
        # print(f"inputs['attention_mask'].shape = {inputs['attention_mask'].shape}")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Print the loss and gradient values
        # print(f"Loss: {loss.item():.4f}")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient({name}): {param.grad.norm(2).item():.4f}")

        # Clip the gradients to prevent exploding gradients - max norm 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    logging.info(f"\nEpoch {epoch+1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")


