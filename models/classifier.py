import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def save_model(self):
        weights_folder = 'model_weights'
        os.makedirs(weights_folder, exist_ok=True)
        timestamp = time.strftime("%m%d%Y_%H%M%S")
        paramcount = sum(p.numel() for p in self.parameters()) // 1000        
        filename = os.path.join(weights_folder, f'classifier_{paramcount}k_{timestamp}.pth')
        model_info = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'state_dict': self.state_dict()
        }
        torch.save(model_info, filename)
        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        weights_folder = 'model_weights'
        filepath = os.path.join(weights_folder, filename)
        model_info = torch.load(filepath)
        model = cls(
            input_dim=model_info['input_dim'],
            hidden_dim=model_info['hidden_dim'],
            output_dim=model_info['output_dim'],
            dropout=model_info['dropout']
        )
        model.load_state_dict(model_info['state_dict'])
        print(f"Model loaded from {filepath}")
        return model