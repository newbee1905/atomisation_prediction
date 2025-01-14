from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional
import numpy as np

class CustomStandardScaler(StandardScaler):
	def fit(self, X, y=None):
		if X.ndim > 2:
			X_reshaped = X.reshape(-1, X.shape[-1])
			X_scaled = super().fit(X_reshaped)
			# X_scaled.reshape(X.shape)
		else:
			return super().fit(X, y)

	def transform(self, X, y=None):
		if X.ndim > 2:
			X_reshaped = X.reshape(-1, X.shape[-1])
			X_scaled = super().transform(X_reshaped)
			return X_scaled.reshape(X.shape)
		else:
			return super().transform(X, y)

@dataclass
class CustomStandardScalerInner:
	_scalars: Optional[np.ndarray] = field(default=None, init=False, repr=False)
	_mean: Optional[np.ndarray] = field(default=None, init=False, repr=False)
	_scale: Optional[np.ndarray] = field(default=None, init=False, repr=False)

	def fit(self, X, y=None):
		self._mean = np.mean(X, axis=(0, 2))
		self._scale = np.std(X, axis=(0, 2))
		self._scalars = 1 / self._scale
		return self

	def transform(self, X):
		X_centered = X - self._mean.reshape(1, -1, 1)
		return X_centered * self._scalars.reshape(1, -1, 1)

class StandardScalerWithResizing(CustomStandardScaler):
	def __init__(self, resizing_technique: str = "padding_with_zeros", target_resize: int = 1, **kwargs):
		super().__init__(**kwargs)
		self.resizing_technique = resizing_technique
		self.target_resize = target_resize
		self.valid_resizings = [
			"histogram_binning",
			"radial_cutoffs",
			"truncated_expansion",
			"padding_with_zeros",
		]
		if self.resizing_technique not in self.valid_resizings:
			raise ValueError(
				f"Invalid resizing technique. Choose from: {', '.join(self.valid_resizings)}"
			)

	@staticmethod
	def resize(resizing_technique: str, X: np.ndarray, target_shape: int, target_resize) -> np.ndarray:
		if resizing_technique == "padding_with_zeros":
			max_size = max(arr.shape[0] for arr in X)

			resized_data = np.array([
				np.pad(
					arr,
					((0, max_size - arr.shape[0]), (0, 0)),
					mode="constant"
				) for arr in X
			])

		elif resizing_technique == "truncated_expansion":
			resized_data = np.array([
				arr[:target_resize] if arr.shape[0] > target_resize else np.pad(
					arr, ((0, target_resize - arr.shape[0]), (0, 0)), mode="constant"
				) for arr in X
			])

			return resized_data
		else:
			raise NotImplementedError(f"Resizing technique '{self.resizing_technique}' is not implemented.")

		return resized_data.reshape(resized_data.shape[0], -1)

	def fit(self, X: np.ndarray, y=None):
		target_shape = X.shape[-1]
		resized_X = StandardScalerWithResizing.resize(self.resizing_technique, X, target_shape, self.target_resize)
		super().fit(resized_X, y)
		return self

	def transform(self, X: np.ndarray, y=None):
		target_shape = X.shape[-1]
		resized_X = StandardScalerWithResizing.resize(self.resizing_technique, X, target_shape, self.target_resize)
		return super().transform(resized_X)

	def fit_transform(self, X: np.ndarray, y=None):
		target_shape = X.shape[-1]
		resized_X = StandardScalerWithResizing.resize(self.resizing_technique, X, target_shape, self.target_resize)
		return super().fit_transform(resized_X, y)
