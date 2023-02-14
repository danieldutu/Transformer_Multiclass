import torch.nn as nn
from transformers import DistilBertModel
import torch


class TransformerModel(nn.Module):
    """Transformer model for text classification

    Args:
        output_dim (int): Number of classes
        hidden_dim (int): Hidden dimension of the transformer
        num_layers (int): Number of transformer layers
        dropout (float): Dropout probability

    Methods:
        forward: Forward pass of the model

    Return
        output: Output of the model

    """

    def __init__(self, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=2048, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        input_ids = inputs['input_ids'].squeeze(1)  # Remove extra dimension
        attention_mask = inputs['attention_mask'].squeeze(1)  # Remove extra dimension

        embedded = self.bert(input_ids, attention_mask=attention_mask)[0]

        # Project the embedded tensor to the desired shape
        # embedded = self.proj(embedded)

        # Pass the embedding tensor through nn.Transformer
        output = self.transformer(embedded, embedded)

        # Transpose the output tensor back to the original shape
        output = output.permute(1, 0, 2)

        # Take the last output from the transformer and pass it through the linear layer
        output = self.fc(output[-1])
        # output = torch.sigmoid(output)

        return output


if __name__ == '__main__':
    num_labels = 5
    model = TransformerModel(num_labels, hidden_dim=768, num_layers=4, dropout=0.1)
    print(model)


