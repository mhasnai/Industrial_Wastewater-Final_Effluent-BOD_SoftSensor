from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TARGET_DEFAULT = "F_BOD"

@dataclass
class ModelConfig:
    predictors: List[str]
    target: str = TARGET_DEFAULT

def build_pipelines(n_features: int) -> Tuple[Pipeline, Pipeline]:
    # Scale all features (tree models don't need it, but harmless for demo)
    pre = ColumnTransformer([("num", StandardScaler(), list(range(n_features)))], remainder="drop")
    ext = Pipeline([("pre", pre), ("model", ExtraTreesRegressor(n_estimators=400, random_state=42))])
    mlp = Pipeline([("pre", pre), ("model", MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42, max_iter=1000))])
    return ext, mlp

def split_X_y(df: pd.DataFrame, predictors: List[str], target: str):
    X = df[predictors].copy()
    y = df[target].copy()
    return X, y