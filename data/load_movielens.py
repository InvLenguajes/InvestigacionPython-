"""
Descarga MovieLens (ml-latest-small) y lo carga en SQL Server.
Ejecutar una sola vez: python data/load_movielens.py
"""

import urllib.request
import zipfile
import io
import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

settings = get_settings()


def get_engine():
    conn_str = (
        f"mssql+pyodbc://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_server}:{settings.db_port}/{settings.db_name}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
    )
    return create_engine(conn_str, fast_executemany=True)


def download_movielens():
    print("Descargando MovieLens...")
    with urllib.request.urlopen(MOVIELENS_URL) as response:
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        with z.open("ml-latest-small/movies.csv") as f:
            movies = pd.read_csv(f)
        with z.open("ml-latest-small/ratings.csv") as f:
            ratings = pd.read_csv(f)
        with z.open("ml-latest-small/tags.csv") as f:
            tags = pd.read_csv(f)

    print(f"  Películas: {len(movies):,}")
    print(f"  Ratings:   {len(ratings):,}")
    print(f"  Tags:      {len(tags):,}")
    return movies, ratings, tags


def prepare(movies, ratings, tags):
    users = pd.DataFrame({"userId": sorted(ratings["userId"].unique())})

    ratings = ratings.copy()
    ratings["ratedAt"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = ratings[["userId", "movieId", "rating", "ratedAt"]]

    tags = tags.copy()
    tags["taggedAt"] = pd.to_datetime(tags["timestamp"], unit="s")
    tags = tags[["userId", "movieId", "tag", "taggedAt"]]

    return movies, users, ratings, tags


def load(engine, movies, users, ratings, tags):
    with engine.connect() as conn:
        print("\nLimpiando tablas...")
        conn.execute(text("DELETE FROM Tags"))
        conn.execute(text("DELETE FROM Ratings"))
        conn.execute(text("DELETE FROM Users"))
        conn.execute(text("DELETE FROM Movies"))
        conn.commit()

    for name, df in [("Movies", movies), ("Users", users), ("Ratings", ratings), ("Tags", tags)]:
        print(f"Cargando {name} ({len(df):,} filas)...", end=" ")
        df.to_sql(name=name, con=engine, if_exists="append", index=False, chunksize=1000)
        print("✓")


def verify(engine):
    print("\nVerificación:")
    with engine.connect() as conn:
        for table in ["Movies", "Users", "Ratings", "Tags"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table:<10} {count:>8,} filas")


if __name__ == "__main__":
    engine = get_engine()
    movies, ratings, tags = download_movielens()
    movies, users, ratings, tags = prepare(movies, ratings, tags)
    load(engine, movies, users, ratings, tags)
    verify(engine)
    print("\n✅ Carga completada.")