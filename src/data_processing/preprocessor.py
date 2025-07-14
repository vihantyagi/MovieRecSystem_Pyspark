# src/data_processing/preprocessor.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, when, isnan, stddev, percentile_approx
from pyspark.ml.feature import StringIndexer
from pyspark.sql.window import Window
import logging
import time
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieLensPreprocessor:
    def __init__(self, spark_session=None):
        self.spark = spark_session or self._create_spark_session()
        self.ratings_df = None
        self.movies_df = None
        self.processed_df = None
        self.stats = None
        
    def _create_spark_session(self):
        """Create optimized Spark session for large dataset"""
        return SparkSession.builder \
            .appName("MovieLens-Preprocessing") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
    
    def load_data(self, ratings_path="/app/data/ml-25m/ratings.csv", 
                  movies_path="/app/data/ml-25m/movies.csv"):
        """Load MovieLens 25M data"""
        logger.info("Loading MovieLens 25M dataset...")
        start_time = time.time()
        
        # Load ratings with optimizations
        self.ratings_df = self.spark.read.csv(
            ratings_path,
            header=True,
            inferSchema=True
        ).repartition(200, "userId")  # Repartition by userId for better performance
        
        # Load movies
        self.movies_df = self.spark.read.csv(
            movies_path,
            header=True,
            inferSchema=True
        )
        
        # Cache for multiple operations
        self.ratings_df.cache()
        self.movies_df.cache()
        
        # Force evaluation
        ratings_count = self.ratings_df.count()
        movies_count = self.movies_df.count()
        
        logger.info(f"Loaded {ratings_count:,} ratings and {movies_count:,} movies")
        logger.info(f"Loading completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def clean_data(self):
        """Clean and validate data"""
        logger.info("Cleaning data...")
        start_time = time.time()
        
        # Remove any null values
        initial_count = self.ratings_df.count()
        self.ratings_df = self.ratings_df.dropna()
        
        # Validate rating values
        self.ratings_df = self.ratings_df.filter(
            (col("rating") >= 0.5) & (col("rating") <= 5.0)
        )
        
        final_count = self.ratings_df.count()
        logger.info(f"Removed {initial_count - final_count:,} invalid rows")
        
        # Ensure correct data types
        self.ratings_df = self.ratings_df \
            .withColumn("userId", col("userId").cast("integer")) \
            .withColumn("movieId", col("movieId").cast("integer")) \
            .withColumn("rating", col("rating").cast("float")) \
            .withColumn("timestamp", col("timestamp").cast("long"))
        
        logger.info(f"Cleaning completed in {time.time() - start_time:.2f} seconds")
        return self
    
    def filter_inactive_users_and_items(self, min_user_ratings=20, min_item_ratings=5):
        """Remove users and items with too few interactions"""
        logger.info(f"Filtering users with < {min_user_ratings} ratings and items with < {min_item_ratings} ratings...")
        start_time = time.time()
        
        initial_count = self.ratings_df.count()
        
        # Filter users
        user_counts = self.ratings_df.groupBy("userId").count()
        active_users = user_counts.filter(col("count") >= min_user_ratings).select("userId")
        
        # Filter items
        item_counts = self.ratings_df.groupBy("movieId").count()
        active_items = item_counts.filter(col("count") >= min_item_ratings).select("movieId")
        
        # Apply filters
        self.processed_df = self.ratings_df \
            .join(active_users, "userId") \
            .join(active_items, "movieId")
        
        # Cache the processed dataframe
        self.processed_df.cache()
        final_count = self.processed_df.count()
        
        logger.info(f"Filtered from {initial_count:,} to {final_count:,} ratings")
        logger.info(f"Removed {initial_count - final_count:,} ratings ({(initial_count - final_count) / initial_count:.1%})")
        logger.info(f"Filtering completed in {time.time() - start_time:.2f} seconds")
        
        return self
    
    def create_user_item_indices(self):
        """Create continuous indices for users and items"""
        logger.info("Creating user and item indices...")
        
        # Create string indexers
        user_indexer = StringIndexer(inputCol="userId", outputCol="user_idx")
        item_indexer = StringIndexer(inputCol="movieId", outputCol="item_idx")
        
        # Fit and transform
        self.processed_df = user_indexer.fit(self.processed_df).transform(self.processed_df)
        self.processed_df = item_indexer.fit(self.processed_df).transform(self.processed_df)
        
        # Convert to integers
        self.processed_df = self.processed_df \
            .withColumn("user_idx", col("user_idx").cast("integer")) \
            .withColumn("item_idx", col("item_idx").cast("integer"))
        
        return self
    
    def compute_statistics(self):
        """Compute and save dataset statistics"""
        logger.info("Computing dataset statistics...")
        
        stats = {}
        
        # Basic counts
        stats['total_ratings'] = self.processed_df.count()
        stats['num_users'] = self.processed_df.select("userId").distinct().count()
        stats['num_items'] = self.processed_df.select("movieId").distinct().count()
        
        # Sparsity
        stats['sparsity'] = 1 - (stats['total_ratings'] / (stats['num_users'] * stats['num_items']))
        
        # Rating statistics
        rating_stats = self.processed_df.select(
            avg("rating").alias("avg_rating"),
            stddev("rating").alias("std_rating")
        ).collect()[0]
        
        stats['avg_rating'] = float(rating_stats['avg_rating'])
        stats['std_rating'] = float(rating_stats['std_rating'])
        
        # User statistics
        user_stats = self.processed_df.groupBy("userId").agg(
            count("rating").alias("num_ratings")
        ).select(
            avg("num_ratings").alias("avg_user_ratings"),
            percentile_approx("num_ratings", 0.5).alias("median_user_ratings")
        ).collect()[0]
        
        stats['avg_ratings_per_user'] = float(user_stats['avg_user_ratings'])
        stats['median_ratings_per_user'] = float(user_stats['median_user_ratings'])
        
        # Item statistics
        item_stats = self.processed_df.groupBy("movieId").agg(
            count("rating").alias("num_ratings")
        ).select(
            avg("num_ratings").alias("avg_item_ratings"),
            percentile_approx("num_ratings", 0.5).alias("median_item_ratings")
        ).collect()[0]
        
        stats['avg_ratings_per_item'] = float(item_stats['avg_item_ratings'])
        stats['median_ratings_per_item'] = float(item_stats['median_item_ratings'])
        
        # Log statistics
        logger.info("\n=== Dataset Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value:,}")
        
        # Store stats in the object
        self.stats = stats
        
        return self  # Return self for method chaining
    
    def split_data(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """Split data into train, validation, and test sets"""
        logger.info(f"Splitting data: {train_ratio:.0%} train, {val_ratio:.0%} val, {1-train_ratio-val_ratio:.0%} test")
        
        # Add random column for splitting
        df_with_rand = self.processed_df.withColumn("rand", col("timestamp") % 100)
        
        train_threshold = train_ratio * 100
        val_threshold = (train_ratio + val_ratio) * 100
        
        train_df = df_with_rand.filter(col("rand") < train_threshold).drop("rand")
        val_df = df_with_rand.filter(
            (col("rand") >= train_threshold) & (col("rand") < val_threshold)
        ).drop("rand")
        test_df = df_with_rand.filter(col("rand") >= val_threshold).drop("rand")
        
        # Cache splits
        train_df.cache()
        val_df.cache()
        test_df.cache()
        
        # Force evaluation and log counts
        train_count = train_df.count()
        val_count = val_df.count()
        test_count = test_df.count()
        
        logger.info(f"Train set: {train_count:,} ratings ({train_count/self.processed_df.count():.1%})")
        logger.info(f"Validation set: {val_count:,} ratings ({val_count/self.processed_df.count():.1%})")
        logger.info(f"Test set: {test_count:,} ratings ({test_count/self.processed_df.count():.1%})")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, output_dir="/app/data/processed"):
        """Save processed data to disk"""
        logger.info(f"Saving processed data to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as Parquet for efficiency
        self.processed_df.write \
            .mode("overwrite") \
            .parquet(os.path.join(output_dir, "ratings_processed"))
        
        # Save movie metadata
        self.movies_df.write \
            .mode("overwrite") \
            .parquet(os.path.join(output_dir, "movies"))
        
        logger.info("Data saved successfully!")
        
        return self

# Main preprocessing script
def main():
    """Run the complete preprocessing pipeline"""
    logger.info("Starting MovieLens 25M preprocessing pipeline...")
    start_time = time.time()
    
    # Create preprocessor
    preprocessor = MovieLensPreprocessor()
    
    # Run pipeline
    preprocessor.load_data() \
        .clean_data() \
        .filter_inactive_users_and_items(min_user_ratings=20, min_item_ratings=5) \
        .create_user_item_indices() \
        .compute_statistics() \
        .save_processed_data()
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data()
    
    # Save splits
    output_dir = "/app/data/processed"
    train_df.write.mode("overwrite").parquet(os.path.join(output_dir, "train"))
    val_df.write.mode("overwrite").parquet(os.path.join(output_dir, "validation"))
    test_df.write.mode("overwrite").parquet(os.path.join(output_dir, "test"))
    
    # Save statistics to JSON file
    if preprocessor.stats:
        import json
        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(preprocessor.stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")
    
    logger.info(f"Total preprocessing time: {time.time() - start_time:.2f} seconds")
    logger.info("Preprocessing pipeline completed successfully!")
    
    # Stop Spark
    preprocessor.spark.stop()

if __name__ == "__main__":
    main()
