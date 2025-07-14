# src/data_processing/explore_data.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, stddev, min, max, desc
import time

def create_spark_session():
    """Create optimized Spark session for ML-25M dataset"""
    return SparkSession.builder \
        .appName("MovieLens-25M-Exploration") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def explore_ml25m():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    print("=== Loading MovieLens 25M Dataset ===")
    start_time = time.time()
    
    # Update paths to match your ml-25m location
    ratings_path = "/app/data/ml-25m/ratings.csv"
    movies_path = "/app/data/ml-25m/movies.csv"
    tags_path = "/app/data/ml-25m/tags.csv"
    
    # Load data with optimizations
    ratings_df = spark.read.csv(
        ratings_path,
        header=True,
        inferSchema=True
    ).cache()  # Cache for multiple operations
    
    movies_df = spark.read.csv(
        movies_path,
        header=True,
        inferSchema=True
    ).cache()
    
    tags_df = spark.read.csv(
        tags_path,
        header=True,
        inferSchema=True
    )
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds\n")
    
    # Basic statistics
    print("=== Dataset Statistics ===")
    print(f"Total ratings: {ratings_df.count():,}")
    print(f"Total movies: {movies_df.count():,}")
    print(f"Total tags: {tags_df.count():,}")
    
    num_users = ratings_df.select("userId").distinct().count()
    num_movies_rated = ratings_df.select("movieId").distinct().count()
    
    print(f"\nUnique users: {num_users:,}")
    print(f"Unique movies rated: {num_movies_rated:,}")
    
    # Calculate sparsity
    total_possible_ratings = num_users * num_movies_rated
    actual_ratings = ratings_df.count()
    sparsity = 1 - (actual_ratings / total_possible_ratings)
    print(f"Matrix sparsity: {sparsity:.4%}")
    
    # Rating distribution
    print("\n=== Rating Distribution ===")
    ratings_df.groupBy("rating") \
        .count() \
        .orderBy("rating") \
        .show()
    
    # Rating statistics
    rating_stats = ratings_df.select(
        avg("rating").alias("avg_rating"),
        stddev("rating").alias("std_rating"),
        min("rating").alias("min_rating"),
        max("rating").alias("max_rating")
    ).collect()[0]
    
    print(f"\nAverage rating: {rating_stats['avg_rating']:.3f}")
    print(f"Std dev of ratings: {rating_stats['std_rating']:.3f}")
    print(f"Min rating: {rating_stats['min_rating']}")
    print(f"Max rating: {rating_stats['max_rating']}")
    
    # User activity analysis
    print("\n=== User Activity Analysis ===")
    user_activity = ratings_df.groupBy("userId") \
        .agg(
            count("rating").alias("num_ratings"),
            avg("rating").alias("avg_rating")
        )
    
    user_stats = user_activity.select(
        avg("num_ratings").alias("avg_ratings_per_user"),
        stddev("num_ratings").alias("std_ratings_per_user"),
        min("num_ratings").alias("min_ratings_per_user"),
        max("num_ratings").alias("max_ratings_per_user")
    ).collect()[0]
    
    print(f"Average ratings per user: {user_stats['avg_ratings_per_user']:.1f}")
    print(f"Std dev: {user_stats['std_ratings_per_user']:.1f}")
    print(f"Min ratings by a user: {user_stats['min_ratings_per_user']}")
    print(f"Max ratings by a user: {user_stats['max_ratings_per_user']}")
    
    # Most active users
    print("\n=== Top 10 Most Active Users ===")
    user_activity.orderBy(desc("num_ratings")) \
        .limit(10) \
        .show()
    
    # Movie popularity analysis
    print("\n=== Movie Popularity Analysis ===")
    movie_popularity = ratings_df.groupBy("movieId") \
        .agg(
            count("rating").alias("num_ratings"),
            avg("rating").alias("avg_rating")
        )
    
    movie_stats = movie_popularity.select(
        avg("num_ratings").alias("avg_ratings_per_movie"),
        stddev("num_ratings").alias("std_ratings_per_movie"),
        min("num_ratings").alias("min_ratings_per_movie"),
        max("num_ratings").alias("max_ratings_per_movie")
    ).collect()[0]
    
    print(f"Average ratings per movie: {movie_stats['avg_ratings_per_movie']:.1f}")
    print(f"Std dev: {movie_stats['std_ratings_per_movie']:.1f}")
    print(f"Min ratings for a movie: {movie_stats['min_ratings_per_movie']}")
    print(f"Max ratings for a movie: {movie_stats['max_ratings_per_movie']}")
    
    # Most popular movies
    print("\n=== Top 10 Most Rated Movies ===")
    popular_movies = movie_popularity \
        .join(movies_df, "movieId") \
        .orderBy(desc("num_ratings")) \
        .select("movieId", "title", "num_ratings", "avg_rating") \
        .limit(10)
    
    popular_movies.show(truncate=False)
    
    # Identify potential issues
    print("\n=== Data Quality Checks ===")
    
    # Check for null values
    null_ratings = ratings_df.filter(
        col("userId").isNull() | 
        col("movieId").isNull() | 
        col("rating").isNull()
    ).count()
    print(f"Null values in ratings: {null_ratings}")
    
    # Check rating range
    invalid_ratings = ratings_df.filter(
        (col("rating") < 0.5) | (col("rating") > 5.0)
    ).count()
    print(f"Invalid ratings (outside 0.5-5.0): {invalid_ratings}")
    
    # Movies with very few ratings (potential cold start items)
    cold_movies = movie_popularity.filter(col("num_ratings") < 5).count()
    print(f"\nMovies with < 5 ratings: {cold_movies:,} ({cold_movies/num_movies_rated:.1%})")
    
    # Users with very few ratings (potential cold start users)
    cold_users = user_activity.filter(col("num_ratings") < 5).count()
    print(f"Users with < 5 ratings: {cold_users:,} ({cold_users/num_users:.1%})")
    
    # Genre analysis
    print("\n=== Genre Distribution ===")
    genres_df = movies_df.select("movieId", "genres")
    
    # Count movies without genres
    no_genre_count = genres_df.filter(
        col("genres").isNull() | (col("genres") == "(no genres listed)")
    ).count()
    print(f"Movies without genres: {no_genre_count}")
    
    spark.stop()
    print(f"\nTotal exploration time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    explore_ml25m()
