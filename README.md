# Movie Recommendation System

A scalable movie recommendation system built with Apache Spark and MLlib using collaborative filtering (ALS algorithm). The system includes a FastAPI backend, Streamlit dashboard, and is fully containerized with Docker.

## Tech Stack

- **Apache Spark 3.5.0** - Distributed data processing and machine learning
- **PySpark** - Python API for Spark
- **MLlib ALS** - Collaborative filtering algorithm for recommendations
- **FastAPI** - REST API backend
- **Streamlit** - Interactive web dashboard
- **Docker & Docker Compose** - Containerization
- **MovieLens 25M Dataset** - 25M ratings across 62K movies

### Key Libraries
- **API**: FastAPI, Uvicorn, Pydantic, NumPy
- **Dashboard**: Streamlit, Pandas, Plotly, Requests
- **ML/Processing**: PySpark, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## Dataset

This project uses the MovieLens 25M dataset containing:
- 25,000,095 ratings
- 62,423 movies
- 162,541 users
- Rating period: 1995-2019

## Setup and Installation

### Prerequisites
- Docker and Docker Compose
- At least 8GB RAM recommended

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MovieRecSystem
   ```

2. **Build and start services**
   ```bash
   docker-compose up --build
   ```

3. **Access the applications**
   - **Dashboard**: http://localhost:8501
   - **API**: http://localhost:8000
   - **Spark UI**: http://localhost:4040

### Training the Model
1. **Download MovieLens 25M Dataset and place it in the data folder**
  
1. **Run Data Preprocessing**
   ```
   docker exec -it pyspark-movie-rec python /app/src/data_processing/ preprocessor.py
   ```

2. **Train the model**
   ```
   docker exec -it pyspark-movie-rec python /app/src/train.py \
    --rank 100 \
    --max-iter 10 \
    --reg-param 0.1 \
    --save-model
   ```
3. **Run Full Evaluation**
   ```
   docker exec -it pyspark-movie-rec python /app/src/run_full_evaluation.py
   ``` 
## Services

### Dashboard Service (Port 8501)
- Streamlit web interface
- Interactive movie recommendations
- Model performance metrics
- User-friendly recommendation interface

## Dashboard 

<img width="1704" height="929" alt="Screenshot 2025-07-15 at 4 06 03 PM" src="https://github.com/user-attachments/assets/43767ca4-1e8d-44b7-9d8f-7dc33b012562" />

<img width="1675" height="956" alt="Screenshot 2025-07-15 at 4 06 29 PM" src="https://github.com/user-attachments/assets/9848442d-1933-468f-aa59-10500d1b4394" />

<img width="1706" height="972" alt="Screenshot 2025-07-15 at 4 06 55 PM" src="https://github.com/user-attachments/assets/efafdbff-fce6-4db9-92e0-810098400906" />

## Architecture

<img width="1087" height="363" alt="Screenshot 2025-07-15 at 4 03 31 PM" src="https://github.com/user-attachments/assets/735b2a12-7149-4f19-b906-5924c281f847" />

## Project Structure

```
MovieRecSystem/
├── src/
│   ├── api/              # FastAPI backend
│   ├── dashboard/        # Streamlit frontend
│   ├── data_processing/  # Data preprocessing
│   ├── model/           # ALS model and evaluation
│   ├── train.py         # Model training script
│   └── run_full_evaluation.py # to get metrics of the trained model. 
├── data/
│   ├── ml-25m/          # MovieLens dataset
│   └── processed/       # Processed data
├── models/              # Trained model artifacts
├── docker/              # Dockerfiles
├── docker-compose.yml
└── requirements-*.txt   # Dependencies
```

## Usage

1. Start the system with `docker-compose up`
2. Open the dashboard at http://localhost:8501
3. Change to API url in the sidebar: http://api:8000
4. Enter a user ID to get personalized movie recommendations
5. Use the API directly at http://localhost:8000/docs for programmatic access

## Model Details

- **Algorithm**: Alternating Least Squares (ALS)
- **Framework**: Apache Spark MLlib
- **Training Data**: MovieLens 25M ratings
- **Evaluation**: RMSE, Precision, Recall metrics
- **Scalability**: Designed for distributed processing
