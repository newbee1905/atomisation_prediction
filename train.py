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
from cuml.common.device_selection import using_device_type

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train ML models for molecular property prediction")

	parser.add_argument(
		"--model", type=str, default="random_forest",
		help="ML model to use (random_forest, ridge, lasso)"
	)
	parser.add_argument(
		"--descriptor", type=str, default="soap",
		help="Molecular descriptor to use (soap, acsf, mbtr, lmbtr, coulomb, sine, ewald_sum)"
	)

	parser.add_argument(
		"--dataset_path", type=str, default="Molecules",
		help="Path to molecule dataset",
	)
	parser.add_argument(
		"--cache_path", type=str, default="atom_ds.pickle",
		help="Path to dataset cache",
	)
	parser.add_argument(
		"--test_size", type=float, default=0.2,
		help="Fraction of data to use for testing",
	)
	parser.add_argument(
		"--random_state", type=int, default=42,
		help="Random seed for reproducibility",
	)
	parser.add_argument(
		"--n_jobs", type=int, default=None,
		help="Number of parallel jobs. Defaults to number of CPU cores",
	)

	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

	logging.info("Loading dataset...")
	dataset = AtomDataset(dataset_path=args.dataset_path, cache_path=args.cache_path)
	try:
		dataset.load_cache()
	except FileNotFoundError:
		dataset.load_dataset()
		dataset.calculate_all_atomisation_energies()
		dataset.save_cache()

	if args.n_jobs is None:
		args.n_jobs = os.cpu_count()

	desc_config = descriptor_configs[args.descriptor]
	model_config = model_configs[args.model]

	# Create descriptor
	logging.info(f"Creating {args.descriptor.upper()} descriptor...")
	descriptor = get_descriptor(
		desc_config, 
		list(dataset.unique_atoms),
		dataset.max_n_atoms,
	)

	X = descriptor.create(dataset.molecules, n_jobs=args.n_jobs)
	y = np.array(list(dataset.atomisation_energies.values()))
	X = np.asarray(X , dtype=np.ndarray)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=args.random_state
	)

	# scaler = CustomStandardScaler()
	scaler = StandardScalerWithResizing("truncated_expansion")
	scaler.fit(X_train)
	X_train_scaled = scaler.transform(X_train)

	logging.info(f"Training {args.model} model...")
	model = get_model(model_config, args.random_state)
	model.fit(X_train_scaled, y_train)

	r2, rmse, acc = validate_model(model, scaler, X_test, y_test)

	logging.info("\nModel Performance:")
	logging.info(f"R2 Score: {r2:.4f}")
	logging.info(f"RMSE: {rmse:.4f} eV")
	logging.info(f"Accuracy: {acc:.4f}")

	save_model(model, scaler, descriptor, args.model, args.descriptor, scaler.resizing_technique)
