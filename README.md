# Movie Recommender — Módulo ML (Python)

Sistema de recomendación de películas basado en **Collaborative Filtering con SVD**.
Este repositorio expone una API REST consumida por el backend Spring Boot.

## Arquitectura general

Angular → Spring Boot → Python ML API → SQL Server

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| Framework API | FastAPI |
| Algoritmo ML | SVD (numpy + scikit-learn) |
| Base de datos | SQL Server |
| ORM / Connector | SQLAlchemy + pyodbc |
| Testing | pytest |

## ¿Cómo funciona el algoritmo?

**Collaborative Filtering con SVD (Singular Value Decomposition)**

El algoritmo aprende a representar usuarios y películas en un espacio de factores latentes — características ocultas que el modelo descubre automáticamente, como preferencia por acción, drama o ciencia ficción.

Para recomendar, se predicen los ratings de películas no vistas y se retornan las de mayor puntaje.

**Métricas obtenidas con MovieLens (100,836 ratings):**
- **RMSE**: 0.6692
- **MAE**: 0.4471

## Setup

### 1. Prerrequisitos

- Python 3.11+
- ODBC Driver 17 for SQL Server

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Completar con las credenciales de SQL Server
```

### 4. Crear el esquema en SQL Server

Ejecutar `data/schema.sql` en SSMS.

### 5. Cargar el dataset MovieLens

```bash
python data/load_movielens.py
```

### 6. Iniciar la API

```bash
python -m uvicorn app.main:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio y la BD |
| POST | `/recommendations/train` | Entrena el modelo SVD |
| GET | `/recommendations/user/{userId}` | Top N recomendaciones para un usuario |
| GET | `/recommendations/predict/{userId}/{movieId}` | Predice el rating de usuario→película |
| GET | `/recommendations/metrics` | Historial de métricas de entrenamiento |

## Tests

```bash
python -m pytest tests/ -v
```

## Estructura del proyecto

├── app/
│   ├── main.py                         # Entry point FastAPI
│   ├── config.py                       # Variables de entorno
│   ├── db/
│   │   └── connection.py               # Conexión SQL Server
│   ├── models/
│   │   └── collaborative_filtering.py  # Algoritmo SVD
│   ├── routers/
│   │   └── recommendations.py          # Endpoints REST
│   └── services/
│       └── recommender_service.py      # Lógica de negocio
├── data/
│   ├── schema.sql                      # Esquema SQL Server
│   └── load_movielens.py               # Carga de MovieLens
├── tests/
│   └── test_model.py                   # Tests unitarios
├── .env.example
└── requirements.txt