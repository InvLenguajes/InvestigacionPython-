import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.collaborative_filtering import (
    CollaborativeFilteringModel,
    Recommendation,
    TrainResult,
)
from app.config import get_settings

settings = get_settings()

_model = CollaborativeFilteringModel(model_path=settings.model_path)


def _fetch_ratings(db: Session) -> pd.DataFrame:
    result = db.execute(text("SELECT userId, movieId, rating FROM Ratings"))
    return pd.DataFrame(result.fetchall(), columns=["userId", "movieId", "rating"])


def _fetch_movies(db: Session) -> pd.DataFrame:
    result = db.execute(text("SELECT movieId, title, genres FROM Movies"))
    return pd.DataFrame(result.fetchall(), columns=["movieId", "title", "genres"])


def _user_exists(db: Session, user_id: int) -> bool:
    result = db.execute(text("SELECT 1 FROM Users WHERE userId = :uid"), {"uid": user_id})
    return result.fetchone() is not None


def _movie_exists(db: Session, movie_id: int) -> bool:
    result = db.execute(text("SELECT 1 FROM Movies WHERE movieId = :mid"), {"mid": movie_id})
    return result.fetchone() is not None


def train_model(db: Session) -> TrainResult:
    ratings_df = _fetch_ratings(db)
    movies_df  = _fetch_movies(db)

    if ratings_df.empty:
        raise ValueError("No hay ratings en la base de datos.")

    result = _model.train(ratings_df, movies_df)

    db.execute(
        text("""
            INSERT INTO ModelMetrics (algorithm, rmse, mae, totalRatings)
            VALUES (:algo, :rmse, :mae, :total)
        """),
        {"algo": "SVD", "rmse": result.rmse, "mae": result.mae, "total": result.total_ratings}
    )
    db.commit()

    return result


def get_recommendations_for_user(user_id: int, db: Session, top_n: int = None) -> list[Recommendation]:
    if not _model.is_trained:
        raise RuntimeError("El modelo no ha sido entrenado. Llamá a POST /recommendations/train primero.")

    if not _user_exists(db, user_id):
        raise ValueError(f"El usuario {user_id} no existe.")

    top_n = top_n or settings.top_n_recommendations
    ratings_df = _fetch_ratings(db)
    movies_df  = _fetch_movies(db)

    return _model.recommend(user_id, movies_df, ratings_df, top_n=top_n)


def get_predicted_rating(user_id: int, movie_id: int, db: Session) -> float:
    if not _model.is_trained:
        raise RuntimeError("El modelo no ha sido entrenado.")

    if not _user_exists(db, user_id):
        raise ValueError(f"El usuario {user_id} no existe.")

    if not _movie_exists(db, movie_id):
        raise ValueError(f"La película {movie_id} no existe.")

    return _model.predict_rating(user_id, movie_id)


def get_model_metrics(db: Session) -> list[dict]:
    result = db.execute(text("""
        SELECT TOP 10 metricId, algorithm, rmse, mae, totalRatings, trainedAt, notes
        FROM ModelMetrics
        ORDER BY trainedAt DESC
    """))
    rows = result.fetchall()
    return [
        {
            "metricId":     row.metricId,
            "algorithm":    row.algorithm,
            "rmse":         float(row.rmse),
            "mae":          float(row.mae),
            "totalRatings": row.totalRatings,
            "trainedAt":    str(row.trainedAt),
            "notes":        row.notes,
        }
        for row in rows
    ]