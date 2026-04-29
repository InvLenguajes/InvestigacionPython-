"""
Collaborative Filtering con SVD implementado con numpy y scikit-learn.

¿Qué es SVD?
    Descompone la matriz usuario-película en factores latentes.
    Permite predecir ratings para pares usuario-película no vistos.
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainResult:
    rmse: float
    mae: float
    total_ratings: int
    model_path: str


@dataclass
class Recommendation:
    movie_id: int
    title: str
    genres: str
    predicted_rating: float


class CollaborativeFilteringModel:

    def __init__(self, model_path: str = "data/svd_model.pkl"):
        self.model_path = model_path
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_index: dict = {}   # userId -> índice de fila
        self.item_index: dict = {}   # movieId -> índice de columna
        self.global_mean: float = 0.0
        self.n_factors: int = 50
        self._load_if_exists()

    # ------------------------------------------------------------------ #
    # Entrenamiento                                                        #
    # ------------------------------------------------------------------ #

    def train(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> TrainResult:
        print(f"[SVD] Entrenando con {len(ratings_df):,} ratings...")

        # 1. Construir índices usuario y película
        users = ratings_df["userId"].unique()
        movies = ratings_df["movieId"].unique()
        self.user_index = {uid: i for i, uid in enumerate(users)}
        self.item_index = {mid: i for i, mid in enumerate(movies)}

        n_users = len(users)
        n_items = len(movies)

        # 2. Construir la matriz usuario-película (sparse → dense)
        #    Las celdas vacías se rellenan con 0 inicialmente
        matrix = np.zeros((n_users, n_items))
        for _, row in ratings_df.iterrows():
            u = self.user_index[row["userId"]]
            i = self.item_index[row["movieId"]]
            matrix[u, i] = float(row["rating"])

        # 3. Calcular la media global y centrar la matriz
        #    Centramos para que SVD capture desviaciones, no valores absolutos
        self.global_mean = ratings_df["rating"].mean()
        matrix_centered = np.where(matrix != 0, matrix - self.global_mean, 0)

        # 4. Aplicar SVD truncado
        #    U: factores de usuario, S: valores singulares, Vt: factores de ítem
        U, S, Vt = np.linalg.svd(matrix_centered, full_matrices=False)

        # 5. Quedarse solo con los primeros n_factors (reducción dimensional)
        k = min(self.n_factors, len(S))
        self.user_factors = U[:, :k] * S[:k]
        self.item_factors = Vt[:k, :].T

        # 6. Evaluar en los ratings conocidos
        predictions, actuals = [], []
        for _, row in ratings_df.iterrows():
            pred = self.predict_rating(int(row["userId"]), int(row["movieId"]))
            predictions.append(pred)
            actuals.append(float(row["rating"]))

        rmse = mean_squared_error(actuals, predictions) ** 0.5
        mae  = mean_absolute_error(actuals, predictions)

        print(f"[SVD] RMSE: {rmse:.4f} | MAE: {mae:.4f}")

        self._save()

        return TrainResult(
            rmse=round(rmse, 4),
            mae=round(mae, 4),
            total_ratings=len(ratings_df),
            model_path=self.model_path
        )

    # ------------------------------------------------------------------ #
    # Predicción y recomendación                                           #
    # ------------------------------------------------------------------ #

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if not self.is_trained:
            raise RuntimeError("El modelo no está entrenado.")

        u = self.user_index.get(user_id)
        i = self.item_index.get(movie_id)

        if u is None or i is None:
            return self.global_mean

        predicted = self.global_mean + self.user_factors[u] @ self.item_factors[i]
        return float(np.clip(predicted, 0.5, 5.0))

    def recommend(
        self,
        user_id: int,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        top_n: int = 10
    ) -> list[Recommendation]:
        if not self.is_trained:
            raise RuntimeError("El modelo no está entrenado.")

        # Películas ya vistas por el usuario
        watched_ids = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())

        # Predecir rating para cada película no vista
        candidates = movies_df[~movies_df["movieId"].isin(watched_ids)]
        predictions = [
            (row["movieId"], row["title"], row["genres"],
             self.predict_rating(user_id, row["movieId"]))
            for _, row in candidates.iterrows()
        ]

        # Ordenar por rating predicho
        predictions.sort(key=lambda x: x[3], reverse=True)

        return [
            Recommendation(
                movie_id=int(mid),
                title=title,
                genres=genres,
                predicted_rating=round(rating, 2)
            )
            for mid, title, genres, rating in predictions[:top_n]
        ]

    # ------------------------------------------------------------------ #
    # Persistencia                                                         #
    # ------------------------------------------------------------------ #

    def _save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"[SVD] Modelo guardado en {self.model_path}")

    def _load_if_exists(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.__dict__.update(pickle.load(f))
            print(f"[SVD] Modelo cargado desde {self.model_path}")

    @property
    def is_trained(self) -> bool:
        return self.user_factors is not None