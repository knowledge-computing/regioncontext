import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from regioncontext.dimension_reducer._base import DimensionReducerBase
from regioncontext.utils import const

import torch.nn as nn
import torch.optim as optim

class AutoencoderReducer(DimensionReducerBase):
    def __init__(self):
        super(AutoencoderReducer, self).__init__()

    def fit_transform(self, csv_file_path, enc_csv_file_path, dimension=64, epoch = 300):
        self.csv_file_path = csv_file_path
        self.enc_csv_file_path = enc_csv_file_path
        self.dimension = dimension 
        self.epoch = epoch

        self.df = pd.read_csv(self.csv_file_path)
    
        embdf = self.df[const.spabert_emb_field_name].apply(lambda x: list(map(float, x.strip('[]').split())))
        embdf = pd.DataFrame(embdf.tolist())

        x_data = torch.tensor(embdf.values, dtype=torch.float32)
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_data_tensor = x_data.to(device)
        x_dataset = TensorDataset(x_data_tensor, x_data_tensor)

        x_loader = DataLoader(x_dataset, batch_size=256, shuffle=True)
       
        autoencoder = Autoencoder(in_shape=x_data.shape[1], enc_shape=self.dimension).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adadelta(autoencoder.parameters())
        train(autoencoder, criterion, optimizer, self.epoch, x_loader, device)
        encoded_data = autoencoder.encode(x_data_tensor)
        encoded_data = encoded_data.cpu()

        encoded_df = pd.DataFrame(encoded_data.detach().numpy())
        self.df[const.spabert_emb_enc_field_name] =''
        self.df[const.spabert_emb_enc_field_name] = encoded_df.apply(lambda x: '[' + ' '.join(x.astype(str)) + ']', axis=1)
        self.df.to_csv(self.enc_csv_file_path, index=False)
        return self.df

def main():
    pass
 # Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

def train(model, criterion, optimizer, n_epochs, train_loader, device):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}")
if __name__ == "__main__":
    main()