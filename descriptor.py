from dataclasses import dataclass
import os
from typing import List, Dict, Any

from dscribe.descriptors import (
	SOAP, ACSF, MBTR, LMBTR, 
	CoulombMatrix, SineMatrix, EwaldSumMatrix
)

@dataclass
class DescriptorConfig:
	name: str
	params: Dict[str, Any]

descriptor_configs = {
	"soap": DescriptorConfig("soap", {
		"r_cut": 6.0,
		"n_max": 8,
		"l_max": 6,
		"sigma": 0.3,
		"periodic": False,
		"sparse": False
	}),
	"acsf": DescriptorConfig("acsf", {
		"r_cut": 6.0,
		"g2_params": [[1, 1], [1, 2], [1, 3]],
		"g4_params": [[1, 1, 1], [1, 2, 1], [1, 1, -1]],
	}),
	"mbtr": DescriptorConfig("mbtr", {
		"geometry": {"function": "distance"},
		"grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
		"weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
		"periodic": False,
		"normalization": "l2",
	}),
	"lmbtr": DescriptorConfig("lmbtr", {
		"geometry": {"function": "distance"},
		"grid": {"min": 0, "max": 5, "n": 100, "sigma": 0.1},
		"weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
		"periodic": False,
		"normalization": "l2",
	}),
	"coulomb": DescriptorConfig("coulomb", {
		"permutation": "sorted_l2",
	}),
	"sine": DescriptorConfig("sine", {
		"permutation": "sorted_l2",
		"sparse": False,
	}),
	"ewald_sum": DescriptorConfig("ewald_sum", {
		"permutation": "sorted_l2",
		"sparse": False,
	})
}


def get_descriptor(config: DescriptorConfig, species: list, max_n_atoms: int):
	descriptors = {
		"coulomb": CoulombMatrix,
		"sine": SineMatrix,
		"ewald_sum": EwaldSumMatrix,
		"soap": SOAP,
		"acsf": ACSF,
		"mbtr": MBTR,
		"lmbtr": LMBTR
	}
	
	if config.name not in descriptors:
		raise ValueError(f"Descriptor {config.name} not supported. Choose from: {list(descriptors.keys())}")
	
	params = config.params.copy()

	if config.name in ["coulomb", "sine", "ewald_sum"]:
		params["n_atoms_max"] = max_n_atoms
	else:
		params["species"] = species
	
	return descriptors[config.name](**params)
