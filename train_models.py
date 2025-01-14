from dataset import AtomDataset
import logging
import os

import argparse
from typing import List, Dict, Any

from model import get_model, model_configs, validate_model, save_model, load_model
from descriptor import get_descriptor, descriptor_configs
from scaler import CustomStandardScaler, CustomStandardScalerInner, StandardScalerWithResizing

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.metrics import r2_score, mean_squared_error, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_models = ["random_forest", "ridge", "lasso"]
# list_descriptors = ["coulomb", "sine", "ewald_sum"]
list_descriptors = ["acsf", "soap", "mbtr", "lmbtr"]
list_resizings = ["truncated_expansion"]

logging.info("Loading dataset...")
dataset = AtomDataset()
try:
	dataset.load_cache()
except FileNotFoundError:
	dataset.load_dataset()
	dataset.calculate_all_atomisation_energies()
	dataset.save_cache()

descriptors = [
	get_descriptor(descriptor_configs[descriptor_name], list(dataset.unique_atoms), dataset.max_n_atoms)
	for descriptor_name in list_descriptors
]

y = np.array(list(dataset.atomisation_energies.values()))

for resizing in list_resizings:
	for i, descriptor in enumerate(descriptors):
		logging.info(f"\n==========================\nDescriptor {list_descriptors[i]}\n==========================\n")
		X = descriptor.create(dataset.molecules, n_jobs=10)
		X = np.asarray(X , dtype=np.ndarray)

		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42
		)

		scaler = StandardScalerWithResizing(resizing)
		scaler.fit(X_train)
		X_train_scaled = scaler.transform(X_train)

		for model_name in list_models:
			logging.info(f"Training {model_name} model...")
			model = get_model(model_configs[model_name], 42)
			model.fit(X_train_scaled, y_train)
			r2, rmse, acc = validate_model(model, scaler, X_test, y_test)

			logging.info(f"\nModel {model_name} Performance:")
			logging.info(f"R2 Score: {r2:.4f}")
			logging.info(f"RMSE: {rmse:.4f} eV")
			logging.info(f"Accuracy: {acc:.4f}")

			save_model(model, scaler, descriptor, model_name, list_descriptors[i], resizing)
