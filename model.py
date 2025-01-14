from dataclasses import dataclass
from cuml.ensemble import RandomForestRegressor
from cuml.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorSK
from sklearn.linear_model import Ridge as RidgeSK, Lasso as LassoSK
from cuml.svm import SVR
from typing import List, Dict, Any,  Optional
from sklearn.preprocessing import StandardScaler
import numpy as np
from cuml.metrics import r2_score, mean_squared_error, accuracy_score
import pickle

@dataclass
class ModelConfig:
	name: str
	params: Dict[str, Any]

model_configs = {
	"random_forest": ModelConfig("random_forest", {
		"n_estimators": 100,
		"max_depth": 15,
	}),
	"random_forest_sk": ModelConfig("random_forest_sk", {
		"n_estimators": 100,
		"max_depth": 15,
	}),
	"ridge": ModelConfig("ridge", {
		"alpha": 1.0,
	}),
	"lasso": ModelConfig("lasso", {
		"alpha": 1.0,
	}),
	"ridge_sk": ModelConfig("ridge_sk", {
		"alpha": 1.0,
	}),
	"lasso_sk": ModelConfig("lasso_sk", {
		"alpha": 1.0,
	}),
}

def get_model(config: ModelConfig, random_state: float):
	models = {
		"random_forest": RandomForestRegressor,
		"ridge": Ridge,
		"lasso": Lasso,
		"random_forest_sk": RandomForestRegressorSK,
		"ridge_sk": RidgeSK,
		"lasso_sk": LassoSK,
	}

	if config.name not in models:
		raise ValueError(f"Model {config.name} not supported. Choose from: {list(models.keys())}")

	params = config.params.copy()

	if config.name != "svr":
		params["random_state"] = random_state
	
	return models[config.name](**params)

def validate_model(model: Any, scaler: StandardScaler, X: np.array, y: np.array):
	X_scaled = scaler.transform(X)
	y_pred = model.predict(X_scaled)
	r2 = r2_score(y, y_pred)
	rmse = np.sqrt(mean_squared_error(y, y_pred))
	acc = accuracy_score(y, y_pred)

	return r2, rmse, acc

def save_model(model: Any, scaler: StandardScaler, descriptor: Any, model_name: str, descriptor_name: str, extra_info: Optional[str] = None):
	model_state_dict = {
		"model": model,
		"scaler": scaler,
		"descriptor": descriptor,
	}

	if extra_info is None:
		model_path = f"models/{model_name}_{descriptor_name}.pickle"
	else:
		model_path = f"models/{model_name}_{descriptor_name}_{extra_info}.pickle"
	with open(model_path, 'wb') as f:
		pickle.dump(model_state_dict, f)

def load_model(model_name: str, descriptor_name: str):
	model_path = f"models/{model_name}_{descriptor_name}.pickle"
	with open(model_path, 'rb') as f:
		return pickle.load(f)

