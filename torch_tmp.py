import os
import logging

from dataset import AtomDataset

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as L

from dscribe.descriptors import SOAP
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RMSELoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.mse = nn.MSELoss()
		
	def forward(self, y_hat, y):
		return torch.sqrt(self.mse(y_hat, y))

class SOAPDataset(Dataset):
	def __init__(self, features, targets):
		self.features = torch.tensor(features)
		self.targets = torch.tensor(targets, dtype=torch.float32)

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		return self.features[idx], self.targets[idx]

dataset = AtomDataset()
try:
	dataset.load_cache()
except FileNotFoundError:
	dataset.load_dataset()
	dataset.calculate_all_atomisation_energies()
	dataset.save_cache()

soap_avg = SOAP(
	r_cut=6.0,
	n_max=8,
	l_max=6,
	sigma=0.3,
	periodic=False,
	sparse=False,
	average="inner",
	dtype="float32",
	species=list(dataset.unique_atoms),
)

soap = SOAP(
	r_cut=6.0,
	n_max=8,
	l_max=6,
	sigma=0.3,
	periodic=False,
	sparse=False,
	dtype="float32",
	species=list(dataset.unique_atoms),
)

n_jobs = os.cpu_count()

X_avg = soap_avg.create(dataset.molecules, n_jobs=n_jobs)
X = soap.create(dataset.molecules, n_jobs=n_jobs)
y = np.array(list(dataset.atomisation_energies.values()))

X_train, X_val, y_train, y_val = train_test_split(
	X, y, test_size=0.2, random_state=42
)

X_train_avg, X_val_avg, _, _ = train_test_split(
	X_avg, y, test_size=0.2, random_state=42
)

max_size = max(arr.shape[0] for arr in X)
X_train_pad = np.array([
	np.pad(
		arr,
		((0, max_size - arr.shape[0]), (0, 0)),
		mode="constant"
	) for arr in X_train
])
X_train_pad_merge = X_train_pad.reshape(X_train_pad.shape[0], -1)
X_val_pad = np.array([
	np.pad(
		arr,
		((0, max_size - arr.shape[0]), (0, 0)),
		mode="constant"
	) for arr in X_val
])
X_val_pad_merge = X_val_pad.reshape(X_val_pad.shape[0], -1)

train_ds_avg = SOAPDataset(X_train_avg, y_train)
val_ds_avg = SOAPDataset(X_val_avg, y_val)

train_ds_pad = SOAPDataset(X_train_pad, y_train)
val_ds_pad = SOAPDataset(X_val_pad, y_val)

train_ds_pad_merge = SOAPDataset(X_train_pad_merge, y_train)
val_ds_pad_merge = SOAPDataset(X_val_pad_merge, y_val)

train_dl_avg = DataLoader(train_ds_avg, batch_size=25, shuffle=True, num_workers=8)
train_dl_pad = DataLoader(train_ds_pad, batch_size=25, shuffle=True, num_workers=8)
train_dl_pad_merge = DataLoader(train_ds_pad_merge, batch_size=25, shuffle=True, num_workers=8)

val_dl_avg = DataLoader(val_ds_avg, batch_size=25, num_workers=8)
val_dl_pad = DataLoader(val_ds_pad, batch_size=25, num_workers=8)
val_dl_pad_merge = DataLoader(val_ds_pad_merge, batch_size=25, num_workers=8)

print(next(iter(train_ds_avg))[0].shape)
print(next(iter(train_ds_pad))[0].shape)
print(next(iter(train_ds_pad_merge))[0].shape)

print(len(train_dl_avg))
print(len(train_dl_pad))
print(len(train_dl_pad_merge))

class SOAPModel(L.LightningModule):
	def __init__(self, model, learning_rate=1e-3):
		super().__init__()
		self.model = model
		self.loss = RMSELoss()
		self.learning_rate = learning_rate

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=64)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		val_loss = self.loss(y_hat, y)
		self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, batch_size=64)
		return val_loss

	def configure_optimizers(self):
		return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

def ff_model_create(input_size, hidden_size, output_size):
	return nn.Sequential(
		nn.Linear(input_size, hidden_size),
		nn.PReLU(),
		nn.Linear(hidden_size, output_size)
	)

class CNNModel(nn.Module):
	def __init__(self, input_channels, output_size, learning_rate=1e-3):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv1d(input_channels, 256, kernel_size=3, dilation=2, padding=2),
			nn.PReLU(),
			nn.Conv1d(256, 512, kernel_size=3, dilation=2, padding=2),
			nn.PReLU(),
			nn.AdaptiveAvgPool1d(1)
		)
		self.fc = nn.Linear(512, output_size)
		self.loss = RMSELoss()
		self.learning_rate = learning_rate

	def forward(self, x):
		x = self.conv(x).squeeze(-1)
		return self.fc(x)

input_size_avg = next(iter(train_ds_avg))[0].shape[0]
input_size_pad_merge = next(iter(train_ds_pad_merge))[0].shape[0]
input_size_pad = next(iter(train_ds_pad))[0].shape[0]
hidden_size = 512
output_size = 1

ff_avg_model = ff_model_create(input_size_avg, hidden_size, output_size)
ff_pad_merge_model = ff_model_create(input_size_pad_merge, hidden_size, output_size)
conv_model = CNNModel(input_size_pad, output_size)

# ff_avg_module = SOAPModel(ff_avg_model)
# avg_trainer = L.Trainer(max_epochs=10)
# avg_trainer.fit(ff_avg_module, train_dl_avg, val_dl_avg)
# torch.save(ff_avg_model.state_dict(), "ff_avg_model.pth")

# ff_pad_merge_module = SOAPModel(ff_pad_merge_model)
# pad_merge_trainer = L.Trainer(max_epochs=10)
# pad_merge_trainer.fit(ff_pad_merge_module, train_dl_pad_merge, val_dl_pad_merge)
# torch.save(ff_pad_merge_model.state_dict(), "ff_pad_merge_model.pth")

conv_module = SOAPModel(conv_model)
conv_trainer = L.Trainer(max_epochs=10)
conv_trainer.fit(conv_module, train_dl_pad, val_dl_pad)
torch.save(conv_model.state_dict(), "conv_model_pad.pth")
