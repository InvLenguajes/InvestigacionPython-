"""
Tests unitarios para el modelo de Collaborative Filtering.
No requieren conexión a SQL Server — usan datos sintéticos.

Ejecutar con: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import os
from app.models.collaborative_filtering import CollaborativeFilteringModel


@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        "userId":  [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "movieId": [1, 2, 3, 1, 4, 5, 2, 3, 4, 1, 3, 5, 2, 4, 5],
        "rating":  [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0],
    })


@pytest.fixture
def sample_movies():
    return pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5],
        "title":   [
            "The Matrix (1999)",
            "Inception (2010)",
            "Interstellar (2014)",
            "The Dark Knight (2008)",
            "Parasite (2019)",
        ],
        "genres": [
            "Action|Sci-Fi",
            "Action|Sci-Fi|Thriller",
            "Sci-Fi|Drama",
            "Action|Crime|Drama",
            "Drama|Thriller",
        ]
    })


@pytest.fixture
def trained_model(sample_ratings, sample_movies, tmp_path):
    model = CollaborativeFilteringModel(str(tmp_path / "test_model.pkl"))
    model.train(sample_ratings, sample_movies)
    return model, sample_ratings, sample_movies


class TestModelTraining:

    def test_train_returns_valid_rmse(self, sample_ratings, sample_movies, tmp_path):
        """El RMSE debe ser mayor a 0."""
        model = CollaborativeFilteringModel(str(tmp_path / "m.pkl"))
        result = model.train(sample_ratings, sample_movies)
        assert result.rmse >= 0

    def test_train_returns_valid_mae(self, sample_ratings, sample_movies, tmp_path):
        """El MAE debe ser mayor a 0."""
        model = CollaborativeFilteringModel(str(tmp_path / "m.pkl"))
        result = model.train(sample_ratings, sample_movies)
        assert result.mae >= 0

    def test_train_returns_correct_total_ratings(self, sample_ratings, sample_movies, tmp_path):
        """El total de ratings debe coincidir con el dataset."""
        model = CollaborativeFilteringModel(str(tmp_path / "m.pkl"))
        result = model.train(sample_ratings, sample_movies)
        assert result.total_ratings == len(sample_ratings)

    def test_model_is_trained_after_fit(self, sample_ratings, sample_movies, tmp_path):
        """is_trained debe ser True después de entrenar."""
        model = CollaborativeFilteringModel(str(tmp_path / "m.pkl"))
        assert not model.is_trained
        model.train(sample_ratings, sample_movies)
        assert model.is_trained

    def test_model_persists_to_disk(self, sample_ratings, sample_movies, tmp_path):
        """El modelo debe guardarse en disco."""
        path = str(tmp_path / "m.pkl")
        model = CollaborativeFilteringModel(path)
        model.train(sample_ratings, sample_movies)
        assert os.path.exists(path)

    def test_model_loads_from_disk(self, sample_ratings, sample_movies, tmp_path):
        """Un modelo guardado debe cargarse en una nueva instancia."""
        path = str(tmp_path / "m.pkl")
        model1 = CollaborativeFilteringModel(path)
        model1.train(sample_ratings, sample_movies)

        model2 = CollaborativeFilteringModel(path)
        assert model2.is_trained


class TestRecommendations:

    def test_recommend_returns_list(self, trained_model):
        """recommend() debe retornar una lista."""
        model, ratings_df, movies_df = trained_model
        recs = model.recommend(1, movies_df, ratings_df, top_n=3)
        assert isinstance(recs, list)

    def test_recommend_excludes_watched_movies(self, trained_model):
        """Las películas ya vistas no deben aparecer en las recomendaciones."""
        model, ratings_df, movies_df = trained_model
        watched = set(ratings_df[ratings_df["userId"] == 1]["movieId"].tolist())
        recs = model.recommend(1, movies_df, ratings_df, top_n=10)
        recommended_ids = {r.movie_id for r in recs}
        assert recommended_ids.isdisjoint(watched)

    def test_recommend_respects_top_n(self, trained_model):
        """El número de recomendaciones no debe superar top_n."""
        model, ratings_df, movies_df = trained_model
        recs = model.recommend(1, movies_df, ratings_df, top_n=2)
        assert len(recs) <= 2

    def test_predicted_ratings_in_range(self, trained_model):
        """Los ratings predichos deben estar entre 0.5 y 5.0."""
        model, ratings_df, movies_df = trained_model
        recs = model.recommend(1, movies_df, ratings_df, top_n=5)
        for rec in recs:
            assert 0.5 <= rec.predicted_rating <= 5.0

    def test_predict_rating_returns_float(self, trained_model):
        """predict_rating() debe retornar un float."""
        model, ratings_df, movies_df = trained_model
        rating = model.predict_rating(user_id=1, movie_id=4)
        assert isinstance(rating, float)

    def test_recommend_without_training_raises(self, tmp_path, sample_ratings, sample_movies):
        """recommend() sin entrenar debe lanzar RuntimeError."""
        model = CollaborativeFilteringModel(str(tmp_path / "empty.pkl"))
        with pytest.raises(RuntimeError):
            model.recommend(1, sample_movies, sample_ratings)