# src/run_full_evaluation.py
import sys
sys.path.append('/app')

from src.model.als_model import MovieRecommender
from src.model.evaluator import RecommenderEvaluator
from pyspark.sql import SparkSession

# Create Spark session with increased memory
spark = SparkSession.builder \
    .appName('Full-Evaluation') \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')

print("Loading model and data...")
recommender = MovieRecommender(spark)
recommender.load_processed_data()
recommender.load_model('/app/models/als_model')

print("Running comprehensive evaluation...")
evaluator = RecommenderEvaluator(spark)
results = evaluator.evaluate_model_comprehensive(
    recommender.model,
    recommender.train_df,
    recommender.val_df,
    recommender.test_df
)

evaluator.save_results()
evaluator.plot_metrics()

print("Evaluation complete!")
print(f"Results saved to /app/models/evaluation_results.json")
spark.stop()