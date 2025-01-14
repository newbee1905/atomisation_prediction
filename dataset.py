from dataclasses import dataclass, field
import pickle
from os import listdir, cpu_count
import logging

from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.atoms import Atoms

from functools import cache
from numba import njit, prange
import numpy as np

import threading
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import List, Set, Dict

REFERENCE_ENERGIES = {
	"H": -0.500,
	"C": -37.846,
	"N": -54.589,
	"O": -75.064,
	"F": -99.733
}

REFERENCE_ENERGIES_ARRAY = np.zeros(100)
for symbol, energy in REFERENCE_ENERGIES.items():
	REFERENCE_ENERGIES_ARRAY[atomic_numbers[symbol]] = energy

@njit
def calculate_atomic_contribution(atomic_numbers_array: np.ndarray, reference_energies_array: np.ndarray) -> float:
	atomic_energy = 0.0

	for atomic_num in atomic_numbers_array:
		atomic_energy += reference_energies_array[atomic_num]

	return atomic_energy


@dataclass
class AtomDataset:
	dataset_path: str = field(default="Molecules")
	cache_path: str = field(default="atom_ds.pickle")
	# dataset_path: str = field(default="MoleculesBig")
	# cache_path: str = field(default="atom_ds_big.pickle")
	unique_atoms: Set[str] = field(default_factory=set)
	molecules: List[Atoms] = field(default_factory=list)
	molecule_energies: List[float] = field(default_factory=list)
	atomisation_energies: Dict[int, float] = field(default_factory=dict)
	max_n_atoms: int = field(default=0, init=False)
	_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

	@staticmethod
	@cache
	def parse_float(s: str):
		s = s.replace("*^", "e")	
		return float(s)

	@staticmethod
	def parse_xyz(file_path):
		with open(file_path) as f:
			n_atom = int(next(f))
			# TODO: read atom value file later
			molecule_info = next(f).split("\t")
			molecule_energy = AtomDataset.parse_float(molecule_info[11])

			symbols, positions = [], []
			for i in range(n_atom):
				atom = next(f).split("\t")
				symbols.append(atom[0])
				positions.append(list(map(AtomDataset.parse_float, atom[1:4])))

			return n_atom, molecule_energy, symbols, positions

	def calculate_atomisation_energy(self, molecule_idx: int) -> float:
		with self._lock:
			if molecule_idx in self.atomisation_energies:
				return self.atomisation_energies[molecule_idx]
		
		molecule = self.molecules[molecule_idx]
		atomic_numbers_array = np.array(molecule.get_atomic_numbers())
		
		atomic_energy = calculate_atomic_contribution(
			atomic_numbers_array,
			REFERENCE_ENERGIES_ARRAY
		)
		
		molecular_energy = self.molecule_energies[molecule_idx]
		
		atomisation_energy = atomic_energy - molecular_energy
		
		with self._lock:
			self.atomisation_energies[molecule_idx] = atomisation_energy
		
		return atomisation_energy

	def _process_batch(self, start_idx: int, end_idx: int) -> None:
		for idx in range(start_idx, end_idx):
			if idx < len(self.molecules):

				self.calculate_atomisation_energy(idx)


	def calculate_all_atomisation_energies(self, num_threads: int = None, batch_size: int = 100) -> None:
		if num_threads is None:
			num_threads = cpu_count()

		total_molecules = len(self.molecules)

		batches = [
			(i, min(i + batch_size, total_molecules))
			for i in range(0, total_molecules, batch_size)
		]

		logging.info(f"Processing {total_molecules} molecules using {num_threads} threads")

		processed_count = 0

		with ThreadPoolExecutor(max_workers=num_threads) as executor:
			future_to_batch = {
				executor.submit(self._process_batch, start, end): (start, end)
				for start, end in batches
			}
			
			for future in as_completed(future_to_batch):
				start, end = future_to_batch[future]

				try:
					future.result()
					processed_count += min(end - start, total_molecules - start)

					print(
						f"Progress: {processed_count}/{total_molecules} molecules processed "
						f"({processed_count/total_molecules*100:.1f}%)"
					)
				except Exception as e:
					logging.error(f"Error processing batch {start}-{end}: {str(e)}")

	def calculate_all_atomisation_energies_sequential(self) -> None:
		for idx in range(len(self.molecules)):
			self.calculate_atomisation_energy(idx)

	def load_dataset(self):
		min_coords = float('inf')
		max_coords = float('-inf')
		for filename in listdir(self.dataset_path):
			filepath = f"{self.dataset_path}/{filename}"
			n_atom, molecule_energy, symbols, positions = AtomDataset.parse_xyz(filepath)

			min_coords = min(min_coords, np.min(positions))
			max_coords = max(max_coords, np.max(positions))

			self.max_n_atoms = max(n_atom, self.max_n_atoms)

			molecule = Atoms(symbols=symbols, positions=positions)
			self.unique_atoms.update(symbols)
			self.molecules.append(molecule)
			self.molecule_energies.append(molecule_energy)

		extent = max_coords - min_coords
		buffer = 3.55
		for mol in self.molecules:
			mol.set_cell([extent + 2 * buffer] * 3)
			mol.set_pbc(False)

	def save_cache(self):
		with open(self.cache_path, "wb") as f:
			self._lock = None
			pickle.dump(self, f)
			self._lock = Lock()

	def load_cache(self):
		with open(self.cache_path, "rb") as f:
			cache = pickle.load(f)

		self.dataset_path = cache.dataset_path
		self.cache_path = cache.cache_path
		self.unique_atoms = cache.unique_atoms
		self.molecules = cache.molecules
		self.molecule_energies = cache.molecule_energies
		self.atomisation_energies = cache.atomisation_energies
		self.max_n_atoms = cache.max_n_atoms
		self._lock = Lock()

	def get_atomisation_statistics(self) -> Dict[str, float]:
		if not self.atomisation_energies:
			self.calculate_all_atomisation_energies()
			
		energies = np.array(list(self.atomisation_energies.values()))
		return {
			"mean": np.mean(energies),
			"std": np.std(energies),
			"min": np.min(energies),
			"max": np.max(energies),
			"median": np.median(energies)
		}

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	dataset = AtomDataset()
	dataset.load_dataset()
	
	# Or load from cache
	# dataset.load_cache()

	dataset.calculate_all_atomisation_energies()

	stats = dataset.get_atomisation_statistics()
	print("\nAtomisation Energy Statistics (eV):")
	for key, value in stats.items():
		print(f"{key.capitalize()}: {value:.2f}")

	dataset.save_cache()
