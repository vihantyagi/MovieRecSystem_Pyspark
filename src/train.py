# src/train.py
import argparse
import logging
import sys
import os

# Add src to path
sys.path.append('/app')

from src.model.als_model import MovieRecommender
from src.model.evaluator import RecommenderEvaluator
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_als_model(args):
    """Main training function"""
    logger.info("Starting ALS model training for MovieLens 25M...")
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ALS-Training-ML25M") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Initialize recommender
        recommender = MovieRecommender(spark)
        
        # Load data
        recommender.load_processed_data(args.data_dir)
        
        # Train model
        recommender.train_model(
            rank=args.rank,
            maxIter=args.max_iter,
            regParam=args.reg_param,
            alpha=args.alpha
        )
        
        # Evaluate on validation set
        val_metrics = recommender.evaluate_model("validation")
        
        # Calculate ranking metrics
        ranking_metrics = recommender.calculate_ranking_metrics(
            k_values=[5, 10, 20],
            rating_threshold=4.0
        )
        
        # Analyze coverage
        coverage = recommender.analyze_coverage(num_recommendations=100)
        
        # Save model
        if args.save_model:
            recommender.save_model(args.model_path)
        
        # Comprehensive evaluation
        if args.full_evaluation:
            logger.info("Running comprehensive evaluation...")
            evaluator = RecommenderEvaluator(spark)
            
            full_results = evaluator.evaluate_model_comprehensive(
                recommender.model,
                recommender.train_df,
                recommender.val_df,
                recommender.test_df
            )
            
            # Save results
            evaluator.save_results()
            
            # Create plots
            if args.create_plots:
                evaluator.plot_metrics()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Train ALS model on MovieLens 25M')
    
    # Data arguments
    parser.add_argument('--data-dir', default='/app/data/processed',
                       help='Directory containing processed data')
    
    # Model hyperparameters
    parser.add_argument('--rank', type=int, default=100,
                       help='Number of latent factors')
    parser.add_argument('--max-iter', type=int, default=10,
                       help='Maximum iterations')
    parser.add_argument('--reg-param', type=float, default=0.1,
                       help='Regularization parameter')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Alpha parameter for implicit feedback')
    
    # Output arguments
    parser.add_argument('--model-path', default='/app/models/als_model',
                       help='Path to save trained model')
    parser.add_argument('--save-model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--full-evaluation', action='store_true',
                       help='Run comprehensive evaluation')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create evaluation plots')
    
    args = parser.parse_args()
    train_als_model(args)

if __name__ == '__main__':
    main()