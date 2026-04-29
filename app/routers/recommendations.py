from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.connection import get_db
from app.services import recommender_service

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


# ------------------------------------------------------------------ #
# Schemas                                                              #
# ------------------------------------------------------------------ #

class RecommendationItem(BaseModel):
    movieId: int
    title: str
    genres: str
    predictedRating: float


class RecommendationsResponse(BaseModel):
    userId: int
    totalRecommendations: int
    recommendations: list[RecommendationItem]


class TrainResponse(BaseModel):
    message: str
    algorithm: str
    rmse: float
    mae: float
    totalRatings: int


class PredictResponse(BaseModel):
    userId: int
    movieId: int
    predictedRating: float


class MetricItem(BaseModel):
    metricId: int
    algorithm: str
    rmse: float
    mae: float
    totalRatings: int
    trainedAt: str
    notes: str | None


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.get("/user/{user_id}", response_model=RecommendationsResponse)
def get_user_recommendations(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    try:
        recs = recommender_service.get_recommendations_for_user(user_id, db, top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return RecommendationsResponse(
        userId=user_id,
        totalRecommendations=len(recs),
        recommendations=[
            RecommendationItem(
                movieId=r.movie_id,
                title=r.title,
                genres=r.genres,
                predictedRating=r.predicted_rating
            )
            for r in recs
        ]
    )


@router.get("/predict/{user_id}/{movie_id}", response_model=PredictResponse)
def predict_rating(user_id: int, movie_id: int, db: Session = Depends(get_db)):
    try:
        rating = recommender_service.get_predicted_rating(user_id, movie_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return PredictResponse(userId=user_id, movieId=movie_id, predictedRating=rating)


@router.post("/train", response_model=TrainResponse)
def train_model(db: Session = Depends(get_db)):
    try:
        result = recommender_service.train_model(db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return TrainResponse(
        message="Modelo entrenado exitosamente.",
        algorithm="SVD",
        rmse=result.rmse,
        mae=result.mae,
        totalRatings=result.total_ratings
    )


@router.get("/metrics", response_model=list[MetricItem])
def get_metrics(db: Session = Depends(get_db)):
    return recommender_service.get_model_metrics(db)