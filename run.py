import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from dataloader import TextLabelDataset
from model import TransformerModel
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

path = r'D:\NLP\nlp-reports-news-classification\water_problem_nlp_en_for_Kaggle_100.csv'

df = pd.read_csv(path, delimiter=';', header=0, encoding='cp1251')

if df.isna().any().any():
    print("DataFrame contains NaN values")
else:
    print("DataFrame does not contain NaN values")
df = df.fillna(0)

train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Set up the data loaders
train_dataset = TextLabelDataset(train_data, max_length=128)
val_dataset = TextLabelDataset(val_data, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Set up the model
model = TransformerModel(output_dim=5, hidden_dim=768, num_layers=4, dropout=0.1)

# Set up the loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set up TensorBoard
writer = SummaryWriter()

# Train the model
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    # Train on the training data
    train_loss = 0
    model.train()
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Clip the gradients to prevent exploding gradients - max norm 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

        # Log the training loss to TensorBoard
        if step % 10 == 0:
            writer.add_scalar('Train/Loss', train_loss / (step + 1), epoch * len(train_loader) + step)

    # Evaluate on the validation data
    val_loss = 0
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            print("Labels: ", labels)

            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    # Print the loss for this epoch
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    logging.info(f"\nEpoch {epoch + 1}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")

    # Log the validation loss to TensorBoard
    writer.add_scalar('Val/Loss', val_loss, epoch)

# Close TensorBoard writer
writer.close()
