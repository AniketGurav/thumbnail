# models3.py
# Mirror of models1.py with the same classifier suite, kept agnostic to the target choice.
# Compatible with main3.py and joint_clean.csv.

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Optional: torch ResNet hook kept as placeholder to stay API-compatible with your earlier design.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.models as tv_models
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class ResNetCustom(nn.Module if _HAS_TORCH else object):
    """(Optional) ResNet18-based regressor. Only built if torch is available."""
    def __init__(self, other_dim: int):
        if not _HAS_TORCH:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.resnet = tv_models.resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()  # 512-D output
        # If you want to concatenate 'other' tabular features, uncomment below and wire forward().
        input_dim = 512  # + other_dim
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, img, other):
        img_feat = self.resnet(img)  # (B, 512)
        x = img_feat  # torch.cat([img_feat, other], dim=1)  # if concatenating tabular
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)


def build_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a family of regressors. The target/metric is decided in main3.py.
    """
    models = {}

    # Linear Regression
    models["Linear Regression"] = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression())
    ])

    # Random Forest
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=config.get("rf_n_estimators", 400),
        max_depth=config.get("rf_max_depth", 14),
        random_state=config.get("random_state", 42),
        n_jobs=-1
    )

    # SVR (RBF)
    models["SVR (RBF)"] = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", SVR(
            kernel="rbf",
            C=config.get("svr_C", 10.0),
            epsilon=config.get("svr_epsilon", 0.2),
            gamma="scale"
        ))
    ])

    # Neural Net (MLP)
    models["Neural Net (MLP)"] = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", MLPRegressor(
            hidden_layer_sizes=config.get("mlp_hidden", (512, 256, 128)),
            activation="relu",
            solver="adam",
            learning_rate_init=config.get("mlp_lr", 1e-3),
            max_iter=config.get("mlp_max_iter", 400),
            random_state=config.get("random_state", 42),
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False
        ))
    ])

    # XGBoost (optional)
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=config.get("xgb_n_estimators", 700),
            learning_rate=config.get("xgb_lr", 0.05),
            max_depth=config.get("xgb_max_depth", 6),
            subsample=config.get("xgb_subsample", 0.8),
            colsample_bytree=config.get("xgb_colsample", 0.8),
            reg_lambda=1.0,
            random_state=config.get("random_state", 42),
            n_jobs=-1
        )
    else:
        print("[models3] xgboost not installed; skipping XGBoost.")

    # Optional: ResNet hook
    if config.get("use_images", False):
        models["ResNet+MLP"] = "custom_pytorch"  # placeholder handled by caller

    return models
