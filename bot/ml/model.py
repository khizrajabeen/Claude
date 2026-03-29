"""Ensemble ML model — XGBoost + LightGBM + LSTM + Transformer.

The ensemble combines tree-based models (strong on tabular features) with
deep learning models (strong on sequential patterns) for robust predictions.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("trading_bot")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  LSTM Network
# ═══════════════════════════════════════════════════════════════

class LSTMNet(nn.Module):
    """Bidirectional LSTM with attention for time-series classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # BUY, HOLD, SELL
        )
        self.regressor = nn.Linear(hidden_dim * 2, 1)  # Return prediction

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        direction = self.classifier(context)  # (batch, 3)
        magnitude = self.regressor(context)   # (batch, 1)
        return direction, magnitude


# ═══════════════════════════════════════════════════════════════
#  Transformer Network
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerPredictor(nn.Module):
    """Transformer encoder for sequential price prediction."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # Use last token
        direction = self.classifier(x)
        magnitude = self.regressor(x)
        return direction, magnitude


# ═══════════════════════════════════════════════════════════════
#  Ensemble Model
# ═══════════════════════════════════════════════════════════════

class EnsembleModel:
    """Ensemble of XGBoost + LightGBM + LSTM + Transformer."""

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.weights = self.model_config.get("ensemble_weights", {
            "xgboost": 0.35, "lightgbm": 0.35, "lstm": 0.20, "transformer": 0.10,
        })
        self.seq_len = self.model_config.get("sequence_length", 60)
        self.top_k = self.model_config.get("features_top_k", 80)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.xgb_model = None
        self.lgb_model = None
        self.lstm_model = None
        self.transformer_model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_trained = False

    def train(self, df: pd.DataFrame, feature_columns: list[str]):
        """Train all models on the feature DataFrame."""
        import xgboost as xgb
        import lightgbm as lgb

        logger.info(f"Training ensemble on {len(df)} samples, {len(feature_columns)} features")
        logger.info(f"Device: {self.device}")

        # Prepare data
        df_clean = df.dropna(subset=feature_columns + ["target_direction"])
        X = df_clean[feature_columns].values
        y_cls = df_clean["target_direction"].values.astype(int)
        y_reg = df_clean["target_return_1"].values.astype(np.float32)

        # Feature selection via XGBoost importance
        logger.info("Running feature selection...")
        temp_xgb = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        X_no_nan = np.nan_to_num(X, nan=0.0)
        temp_xgb.fit(X_no_nan, y_cls)
        importances = temp_xgb.feature_importances_
        top_idx = np.argsort(importances)[-self.top_k :]
        self.feature_columns = [feature_columns[i] for i in top_idx]
        logger.info(f"Selected top {len(self.feature_columns)} features")

        # Re-extract with selected features
        X = df_clean[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = list(tscv.split(X_scaled))[-1]  # Use last split

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_cls_train, y_cls_val = y_cls[train_idx], y_cls[val_idx]
        y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]

        # ── Train XGBoost ──
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=50, verbosity=0, n_jobs=-1,
        )
        self.xgb_model.fit(
            X_train, y_cls_train,
            eval_set=[(X_val, y_cls_val)],
            verbose=False,
        )
        xgb_acc = (self.xgb_model.predict(X_val) == y_cls_val).mean()
        logger.info(f"XGBoost val accuracy: {xgb_acc:.4f}")

        # ── Train LightGBM ──
        logger.info("Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=-1, verbose=-1,
        )
        self.lgb_model.fit(
            X_train, y_cls_train,
            eval_set=[(X_val, y_cls_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        lgb_acc = (self.lgb_model.predict(X_val) == y_cls_val).mean()
        logger.info(f"LightGBM val accuracy: {lgb_acc:.4f}")

        # ── Train LSTM ──
        logger.info("Training LSTM...")
        self.lstm_model = LSTMNet(
            input_dim=len(self.feature_columns), hidden_dim=128, num_layers=2,
        ).to(self.device)
        self._train_neural(
            self.lstm_model, X_scaled, y_cls, y_reg, train_idx, val_idx,
            epochs=100, lr=0.001, name="LSTM",
        )

        # ── Train Transformer ──
        logger.info("Training Transformer...")
        self.transformer_model = TransformerPredictor(
            input_dim=len(self.feature_columns), d_model=64, nhead=4, num_layers=3,
        ).to(self.device)
        self._train_neural(
            self.transformer_model, X_scaled, y_cls, y_reg, train_idx, val_idx,
            epochs=80, lr=0.0005, name="Transformer",
        )

        self.is_trained = True
        self.save()
        logger.info("Ensemble training complete")

    def _train_neural(self, model, X_scaled, y_cls, y_reg, train_idx, val_idx,
                      epochs: int, lr: float, name: str):
        """Train a neural network with sequence data."""
        # Build sequences
        X_seq, y_c_seq, y_r_seq = self._build_sequences(X_scaled, y_cls, y_reg)

        if len(X_seq) == 0:
            logger.warning(f"Not enough data for {name} sequences, skipping")
            return

        # Adjust indices for sequence offset
        max_train = min(len(train_idx), len(X_seq))
        max_val = min(len(val_idx), len(X_seq))
        split_point = max_train

        X_train_t = torch.FloatTensor(X_seq[:split_point]).to(self.device)
        y_c_train = torch.LongTensor(y_c_seq[:split_point]).to(self.device)
        y_r_train = torch.FloatTensor(y_r_seq[:split_point]).to(self.device)

        X_val_t = torch.FloatTensor(X_seq[split_point:split_point + max_val]).to(self.device)
        y_c_val = torch.LongTensor(y_c_seq[split_point:split_point + max_val]).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        cls_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.MSELoss()

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            # Mini-batch
            batch_size = 256
            total_loss = 0
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i + batch_size]
                batch_yc = y_c_train[i:i + batch_size]
                batch_yr = y_r_train[i:i + batch_size]

                direction, magnitude = model(batch_X)
                loss = cls_criterion(direction, batch_yc) + 0.5 * reg_criterion(
                    magnitude.squeeze(), batch_yr
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # Validate
            if len(X_val_t) > 0:
                model.eval()
                with torch.no_grad():
                    val_dir, _ = model(X_val_t)
                    val_pred = val_dir.argmax(dim=1)
                    val_acc = (val_pred == y_c_val).float().mean().item()

                scheduler.step(1 - val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        break

                if (epoch + 1) % 20 == 0:
                    logger.info(
                        f"{name} epoch {epoch+1}/{epochs} | "
                        f"loss={total_loss:.4f} | val_acc={val_acc:.4f}"
                    )

        logger.info(f"{name} best val accuracy: {best_val_acc:.4f}")

    def _build_sequences(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray):
        """Build overlapping sequences for LSTM/Transformer."""
        sequences, labels_cls, labels_reg = [], [], []
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len : i])
            labels_cls.append(y_cls[i])
            labels_reg.append(y_reg[i])
        return np.array(sequences), np.array(labels_cls), np.array(labels_reg)

    def predict(self, df: pd.DataFrame) -> dict:
        """Generate ensemble prediction.

        Returns:
            dict with keys: direction (1=BUY, 0=SELL), confidence, predicted_return,
                            sub_predictions (per-model breakdown)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet — run train() first")

        X_raw = df[self.feature_columns].values
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        X_scaled = self.scaler.transform(X_raw)

        sub = {}

        # XGBoost
        xgb_proba = self.xgb_model.predict_proba(X_scaled[-1:])
        sub["xgboost"] = {"buy_prob": float(xgb_proba[0][1])}

        # LightGBM
        lgb_proba = self.lgb_model.predict_proba(X_scaled[-1:])
        sub["lightgbm"] = {"buy_prob": float(lgb_proba[0][1])}

        # LSTM
        if self.lstm_model and len(X_scaled) >= self.seq_len:
            seq = torch.FloatTensor(X_scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
            self.lstm_model.eval()
            with torch.no_grad():
                direction, magnitude = self.lstm_model(seq)
                probs = torch.softmax(direction, dim=1)[0]
                sub["lstm"] = {
                    "buy_prob": float(probs[1]),
                    "predicted_return": float(magnitude[0]),
                }
        else:
            sub["lstm"] = {"buy_prob": 0.5, "predicted_return": 0.0}

        # Transformer
        if self.transformer_model and len(X_scaled) >= self.seq_len:
            seq = torch.FloatTensor(X_scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
            self.transformer_model.eval()
            with torch.no_grad():
                direction, magnitude = self.transformer_model(seq)
                probs = torch.softmax(direction, dim=1)[0]
                sub["transformer"] = {
                    "buy_prob": float(probs[1]),
                    "predicted_return": float(magnitude[0]),
                }
        else:
            sub["transformer"] = {"buy_prob": 0.5, "predicted_return": 0.0}

        # Weighted ensemble
        w = self.weights
        ensemble_buy_prob = (
            w.get("xgboost", 0) * sub["xgboost"]["buy_prob"]
            + w.get("lightgbm", 0) * sub["lightgbm"]["buy_prob"]
            + w.get("lstm", 0) * sub["lstm"]["buy_prob"]
            + w.get("transformer", 0) * sub["transformer"]["buy_prob"]
        )

        direction = 1 if ensemble_buy_prob > 0.5 else 0
        confidence = abs(ensemble_buy_prob - 0.5) * 2  # 0 to 1 scale

        # Average predicted return from deep models
        predicted_return = (
            sub["lstm"].get("predicted_return", 0) * 0.6
            + sub["transformer"].get("predicted_return", 0) * 0.4
        )

        return {
            "direction": direction,
            "confidence": confidence,
            "buy_probability": ensemble_buy_prob,
            "predicted_return": predicted_return,
            "sub_predictions": sub,
        }

    def save(self):
        """Save all models to disk."""
        if self.xgb_model:
            joblib.dump(self.xgb_model, MODEL_DIR / "xgb_model.pkl")
        if self.lgb_model:
            joblib.dump(self.lgb_model, MODEL_DIR / "lgb_model.pkl")
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), MODEL_DIR / "lstm_model.pt")
        if self.transformer_model:
            torch.save(self.transformer_model.state_dict(), MODEL_DIR / "transformer_model.pt")
        joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(self.feature_columns, MODEL_DIR / "feature_columns.pkl")
        logger.info("Models saved to disk")

    def load(self):
        """Load models from disk."""
        try:
            self.xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")
            self.lgb_model = joblib.load(MODEL_DIR / "lgb_model.pkl")
            self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            self.feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

            n_features = len(self.feature_columns)

            # Load LSTM
            self.lstm_model = LSTMNet(input_dim=n_features).to(self.device)
            self.lstm_model.load_state_dict(
                torch.load(MODEL_DIR / "lstm_model.pt", map_location=self.device, weights_only=True)
            )

            # Load Transformer
            self.transformer_model = TransformerPredictor(input_dim=n_features).to(self.device)
            self.transformer_model.load_state_dict(
                torch.load(MODEL_DIR / "transformer_model.pt", map_location=self.device, weights_only=True)
            )

            self.is_trained = True
            logger.info(f"Models loaded ({n_features} features)")
        except FileNotFoundError as e:
            logger.warning(f"Could not load models: {e}")
            self.is_trained = False
