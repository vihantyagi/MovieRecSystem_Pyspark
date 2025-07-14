# src/model/als_model.py
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, array_contains, count, avg, lit, size, array_intersect
import logging
import time
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, spark_session=None):
        self.spark = spark_session or self._create_spark_session()
        self.model = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.training_history = []
        
    def _create_spark_session(self):
        return SparkSession.builder \
            .appName("MovieRecommender-ALS") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.driver.memory", "6g") \
            .config("spark.executor.memory", "6g") \
            .config("spark.driver.maxResultSize", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.network.timeout", "800s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .getOrCreate()
    
    def load_processed_data(self, data_dir="/app/data/processed"):
        """Load preprocessed data"""
        logger.info(f"Loading processed data from {data_dir}")
        
        self.train_df = self.spark.read.parquet(os.path.join(data_dir, "train"))
        self.val_df = self.spark.read.parquet(os.path.join(data_dir, "validation"))
        self.test_df = self.spark.read.parquet(os.path.join(data_dir, "test"))
        
        # Cache for performance
        self.train_df.cache()
        self.val_df.cache()
        self.test_df.cache()
        
        # Force evaluation
        train_count = self.train_df.count()
        val_count = self.val_df.count()
        test_count = self.test_df.count()
        
        logger.info(f"Loaded train: {train_count:,}, val: {val_count:,}, test: {test_count:,}")
        
        return self
    
    def train_model(self, **als_params):
        """Train ALS model with given parameters"""
        # Default ALS parameters optimized for ML-25M
        params = {
            'rank': 100,
            'maxIter': 10,
            'regParam': 0.1,
            'alpha': 1.0,
            'userCol': 'user_idx',
            'itemCol': 'item_idx',
            'ratingCol': 'rating',
            'coldStartStrategy': 'drop',
            'implicitPrefs': False,
            'seed': 42,
            'checkpointInterval': 10,
            'intermediateStorageLevel': 'MEMORY_AND_DISK',
            'finalStorageLevel': 'MEMORY_AND_DISK'
        }
        params.update(als_params)
        
        logger.info(f"Training ALS model with parameters: {params}")
        
        # Build ALS model
        als = ALS(**params)
        
        # Set checkpoint directory for fault tolerance
        self.spark.sparkContext.setCheckpointDir('/tmp/spark-checkpoint')
        
        # Train model
        start_time = time.time()
        self.model = als.fit(self.train_df)
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Store training info
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'training_time': training_time
        })
        
        return self
    
    def evaluate_model(self, dataset_name="validation", df=None):
        """Evaluate model performance"""
        if df is None:
            df = self.val_df if dataset_name == "validation" else self.test_df
            
        logger.info(f"Evaluating on {dataset_name} set...")
        
        # Make predictions
        predictions = self.model.transform(df)
        predictions.cache()
        
        # Calculate metrics
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        rmse = evaluator.evaluate(predictions)
        
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        
        # Calculate additional metrics
        metrics = {
            'dataset': dataset_name,
            'rmse': rmse,
            'mae': mae
        }
        
        # Calculate prediction statistics
        pred_stats = predictions.select(
            avg("prediction").alias("avg_prediction"),
            count("prediction").alias("num_predictions")
        ).collect()[0]
        
        metrics['avg_prediction'] = float(pred_stats['avg_prediction'])
        metrics['num_predictions'] = pred_stats['num_predictions']
        
        logger.info(f"{dataset_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Unpersist predictions
        predictions.unpersist()
        
        return metrics
    
    def calculate_ranking_metrics(self, k_values=[5, 10, 20], rating_threshold=4.0):
        """Calculate precision@k and recall@k metrics"""
        logger.info(f"Calculating ranking metrics for k={k_values}")
        
        results = {}
        
        for k in k_values:
            # Get actual high-rated items for each user in test set
            actual_df = self.test_df.filter(col("rating") >= rating_threshold) \
                .groupBy("user_idx") \
                .agg(collect_list("item_idx").alias("actual_items"))
            
            # Get predicted top-k items
            predicted_df = self.model.recommendForAllUsers(k) \
                .select(
                    col("user_idx"),
                    col("recommendations.item_idx").alias("predicted_items")
                )
            
            # Join and calculate metrics using Spark SQL functions
            from pyspark.sql.functions import size, array_intersect
            
            metrics_df = actual_df.join(predicted_df, "user_idx", "inner") \
                .select(
                    size(array_intersect("actual_items", "predicted_items")).alias("hits"),
                    size("predicted_items").alias("n_predicted"),
                    size("actual_items").alias("n_actual")
                ) \
                .selectExpr(
                    "CASE WHEN n_predicted > 0 THEN hits / n_predicted ELSE 0 END as precision",
                    "CASE WHEN n_actual > 0 THEN hits / n_actual ELSE 0 END as recall"
                )
            
            # Get average metrics
            avg_metrics = metrics_df.agg(
                avg("precision").alias("avg_precision"),
                avg("recall").alias("avg_recall")
            ).collect()[0]
            
            results[f'precision@{k}'] = float(avg_metrics['avg_precision'])
            results[f'recall@{k}'] = float(avg_metrics['avg_recall'])
            
            logger.info(f"Precision@{k}: {results[f'precision@{k}']:.4f}, " +
                       f"Recall@{k}: {results[f'recall@{k}']:.4f}")
        
        return results
    
    def analyze_coverage(self, num_recommendations=100):
        """Analyze item coverage of recommendations"""
        logger.info(f"Analyzing coverage with {num_recommendations} recommendations per user")
        
        # Get sample of users for coverage analysis (to save computation)
        sample_users = self.train_df.select("user_idx").distinct().sample(0.1, seed=42)
        
        # Get recommendations for sampled users
        all_recs = self.model.recommendForUserSubset(sample_users, num_recommendations)
        
        # Extract all recommended items
        recommended_items = all_recs \
            .select(explode(col("recommendations.item_idx")).alias("item_idx")) \
            .distinct()
        
        # Count unique items
        total_items = self.train_df.select("item_idx").distinct().count()
        recommended_count = recommended_items.count()
        
        coverage = recommended_count / total_items
        logger.info(f"Item coverage: {coverage:.2%} ({recommended_count:,}/{total_items:,})")
        
        return coverage
    
    def get_model_factors_info(self):
        """Get information about learned factors"""
        info = {
            'num_user_factors': self.model.userFactors.count(),
            'num_item_factors': self.model.itemFactors.count(),
            'rank': self.model.rank
        }
        
        # Sample factor statistics
        user_factors_stats = self.model.userFactors.select(
            avg(col("features").getItem(0)).alias("avg_first_factor")
        ).collect()[0]
        
        info['sample_user_factor_mean'] = float(user_factors_stats['avg_first_factor'])
        
        return info
    
    def save_model(self, path="/app/models/als_model"):
        """Save trained model"""
        logger.info(f"Saving model to {path}")
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.write().overwrite().save(path)
        
        # Save training history and metadata
        metadata = {
            'training_history': self.training_history,
            'model_info': self.get_model_factors_info(),
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = path + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model and metadata saved successfully")
        
    def load_model(self, path="/app/models/als_model"):
        """Load pre-trained model"""
        from pyspark.ml.recommendation import ALSModel
        logger.info(f"Loading model from {path}")
        
        self.model = ALSModel.load(path)
        
        # Load metadata if exists
        metadata_path = path + "_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_history = metadata.get('training_history', [])
        
        logger.info("Model loaded successfully")