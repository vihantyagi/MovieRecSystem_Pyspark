# src/api/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import os
import time
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    
class MovieRecommendation(BaseModel):
    movie_id: int
    predicted_rating: float
    rank: int
    title: Optional[str] = None
    original_movie_id: Optional[int] = None
    
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    timestamp: str
    model_version: str = "als_model_v1"
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    last_request: Optional[str]

class ModelInfo(BaseModel):
    model_type: str = "ALS (Alternating Least Squares)"
    dataset: str = "MovieLens 25M"
    num_users: int
    num_items: int
    rank: int
    training_rmse: float
    test_rmse: float

# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Production-ready API serving movie recommendations using ALS collaborative filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_manager = None
start_time = time.time()
last_request_time = None

class ModelManager:
    """Manages the Spark session and ALS model"""
    
    def __init__(self):
        self.spark = None
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = None
        self.item_mapping = None
        self.movies_df = None  # Add this
        self.movie_titles = {}  # Add this
        self.is_loaded = False
        
    def initialize(self):
        """Initialize Spark and load model"""
        logger.info("Initializing ModelManager...")
        
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALSModel
        
        # Create Spark session
        self.spark = SparkSession.builder \
            .appName("MovieRecommendationAPI") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Load model
        model_path = os.getenv("MODEL_PATH", "/app/models/als_model")
        try:
            self.model = ALSModel.load(model_path)
            
            # Cache user and item factors
            self.user_factors = self.model.userFactors.collect()
            self.item_factors = self.model.itemFactors.collect()
            
            # Create mappings
            self.user_mapping = {row['id']: row['features'] for row in self.user_factors}
            self.item_mapping = {row['id']: row['features'] for row in self.item_factors}
            
            # Load movie titles
            self._load_movie_titles()
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Users: {len(self.user_mapping)}, Items: {len(self.item_mapping)}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_movie_titles(self):
        """Load movie titles from processed data"""
        try:
            # Load from processed parquet files
            movies_path = "/app/data/processed/movies"
            movies_df = self.spark.read.parquet(movies_path)
            
            # Create id to title mapping
            # Note: item_idx is from preprocessing, movieId is original
            movies_data = movies_df.select("movieId", "title").collect()
            
            # Also load the mapping between movieId and item_idx
            ratings_path = "/app/data/processed/ratings_processed"
            ratings_df = self.spark.read.parquet(ratings_path)
            
            # Get unique movie mappings
            movie_mapping = ratings_df.select("movieId", "item_idx").distinct().collect()
            
            # Create movieId to title mapping first
            id_to_title = {row['movieId']: row['title'] for row in movies_data}
            
            # Then create item_idx to title mapping
            self.movie_titles = {}
            for row in movie_mapping:
                movie_id = row['movieId']
                item_idx = row['item_idx']
                if movie_id in id_to_title:
                    self.movie_titles[item_idx] = id_to_title[movie_id]
            
            logger.info(f"Loaded titles for {len(self.movie_titles)} movies")
            
        except Exception as e:
            logger.warning(f"Could not load movie titles: {e}")
            # Continue without titles
            self.movie_titles = {}

    def get_recommendations(self, user_id: int, num_recommendations: int) -> List[Dict]:
        """Get recommendations for a user using cached factors"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Check if user exists
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not found in model")
        
        # Get user factors
        user_features = np.array(self.user_mapping[user_id])
        
        # Calculate scores for all items
        scores = []
        for item_id, item_features in self.item_mapping.items():
            score = np.dot(user_features, np.array(item_features))
            scores.append((item_id, float(score)))
        
        # Sort by score and get top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = scores[:num_recommendations]
        
        # Format recommendations with titles
        recommendations = []
        for rank, (item_id, score) in enumerate(top_items, 1):
            rec = {
                "movie_id": item_id,
                "predicted_rating": min(5.0, max(0.5, score)),
                "rank": rank
            }
            
            # Add title if available
            if item_id in self.movie_titles:
                rec["title"] = self.movie_titles[item_id]
                rec["original_movie_id"] = item_id  # This is actually item_idx
            
            recommendations.append(rec)
        
        return recommendations

    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        return {
            "num_users": len(self.user_mapping),
            "num_items": len(self.item_mapping),
            "rank": self.model.rank,
            "training_rmse": 0.8104,  # From your evaluation
            "test_rmse": 0.8121       # From your evaluation
        }

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_manager
    logger.info("Starting up Movie Recommendation API...")
    
    model_manager = ModelManager()
    model_manager.initialize()
    
    logger.info("API startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global model_manager
    if model_manager and model_manager.spark:
        model_manager.spark.stop()
        logger.info("Spark session closed")

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global last_request_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded if model_manager else False,
        uptime_seconds=time.time() - start_time,
        last_request=last_request_time.isoformat() if last_request_time else None
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user"""
    global last_request_time
    last_request_time = datetime.now()
    
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get recommendations
        start = time.time()
        recommendations = model_manager.get_recommendations(
            request.user_id,
            request.num_recommendations
        )
        inference_time = time.time() - start
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {request.user_id} "
                   f"in {inference_time:.3f}s")
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations_by_id(
    user_id: int,
    n: int = Query(10, ge=1, le=100, description="Number of recommendations")
):
    """Get recommendations for a user (GET endpoint)"""
    request = RecommendationRequest(user_id=user_id, num_recommendations=n)
    return await get_recommendations(request)

@app.post("/recommend/batch")
async def get_batch_recommendations(
    user_ids: List[int],
    num_recommendations: int = 10
):
    """Get recommendations for multiple users"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(user_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 users per batch request")
    
    results = []
    for user_id in user_ids:
        try:
            recommendations = model_manager.get_recommendations(user_id, num_recommendations)
            results.append({
                "user_id": user_id,
                "recommendations": recommendations,
                "status": "success"
            })
        except ValueError:
            results.append({
                "user_id": user_id,
                "recommendations": [],
                "status": "user_not_found"
            })
    
    return {"results": results, "timestamp": datetime.now().isoformat()}

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model_manager.get_model_info()
    return ModelInfo(**info)

@app.get("/users/random")
async def get_random_users(n: int = Query(5, ge=1, le=20)):
    """Get random user IDs for testing"""
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    user_ids = list(model_manager.user_mapping.keys())
    import random
    random_users = random.sample(user_ids, min(n, len(user_ids)))
    
    return {
        "user_ids": random_users,
        "total_users": len(user_ids)
    }

# Optional: Add metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "uptime_seconds": time.time() - start_time,
        "model_loaded": model_manager.is_loaded if model_manager else False,
        "last_request": last_request_time.isoformat() if last_request_time else None,
        "version": "1.0.0"
    }