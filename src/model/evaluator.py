# src/model/evaluator.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, stddev, collect_list, size, array_intersect, explode
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Dict

class RecommenderEvaluator:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.results = {}
        
    def evaluate_model_comprehensive(self, model, train_df, val_df, test_df) -> Dict:
        """Comprehensive model evaluation"""
        print("Starting comprehensive evaluation...")
        
        # Basic prediction metrics on validation set
        from pyspark.ml.evaluation import RegressionEvaluator
        
        predictions = model.transform(val_df)
        
        evaluator = RegressionEvaluator(
            labelCol="rating",
            predictionCol="prediction"
        )
        
        # RMSE and MAE
        self.results['val_rmse'] = evaluator.setMetricName("rmse").evaluate(predictions)
        self.results['val_mae'] = evaluator.setMetricName("mae").evaluate(predictions)
        
        # Test set metrics
        test_predictions = model.transform(test_df)
        self.results['test_rmse'] = evaluator.setMetricName("rmse").evaluate(test_predictions)
        self.results['test_mae'] = evaluator.setMetricName("mae").evaluate(test_predictions)
        
        print(f"Validation RMSE: {self.results['val_rmse']:.4f}")
        print(f"Test RMSE: {self.results['test_rmse']:.4f}")
        
        # Ranking metrics
        for k in [5, 10, 20]:
            precision, recall = self._calculate_precision_recall_at_k(
                model, test_df, k=k, threshold=4.0
            )
            self.results[f'precision@{k}'] = precision
            self.results[f'recall@{k}'] = recall
            print(f"Precision@{k}: {precision:.4f}, Recall@{k}: {recall:.4f}")
        
        # Coverage analysis
        self.results['coverage@100'] = self._calculate_coverage(model, train_df, k=100)
        print(f"Coverage@100: {self.results['coverage@100']:.2%}")
        
        # Popularity bias
        self.results['popularity_bias'] = self._analyze_popularity_bias(
            model, train_df, test_df
        )
        
        # Cold start analysis
        self._analyze_cold_start_performance(model, train_df, test_df)
        
        return self.results
    
    def _calculate_precision_recall_at_k(self, model, test_df, k=10, threshold=4.0):
        """Calculate precision and recall at k"""
        # Get relevant items
        relevant_items = test_df.filter(col("rating") >= threshold) \
            .groupBy("user_idx") \
            .agg(collect_list("item_idx").alias("relevant_items"))
        
        # Get recommendations
        # Sample users to speed up evaluation
        test_users = test_df.select("user_idx").distinct().sample(0.1, seed=42)
        recommendations = model.recommendForUserSubset(test_users, k)
        
        # Join and calculate
        joined = relevant_items.join(
            recommendations.select(
                col("user_idx"),
                col("recommendations.item_idx").alias("recommended_items")
            ),
            "user_idx",
            "inner"
        )
        
        # Calculate metrics
        metrics = joined.select(
            size(array_intersect("relevant_items", "recommended_items")).alias("hits"),
            size("recommended_items").alias("n_recommended"),
            size("relevant_items").alias("n_relevant")
        ).agg(
            avg(col("hits") / col("n_recommended")).alias("precision"),
            avg(col("hits") / col("n_relevant")).alias("recall")
        ).collect()[0]
        
        return float(metrics["precision"]), float(metrics["recall"])
    
    def _calculate_coverage(self, model, train_df, k=100):
        """Calculate catalog coverage"""
        # Sample users for efficiency
        sample_users = train_df.select("user_idx").distinct().sample(0.05, seed=42)
        
        # Get recommendations
        recommendations = model.recommendForUserSubset(sample_users, k)
        
        # Count unique recommended items
        unique_recommendations = recommendations \
            .select(explode(col("recommendations.item_idx")).alias("item_idx")) \
            .distinct() \
            .count()
        
        # Total items
        total_items = train_df.select("item_idx").distinct().count()
        
        return unique_recommendations / total_items
    
    def _analyze_popularity_bias(self, model, train_df, test_df):
        """Analyze if model has popularity bias"""
        # Calculate item popularity
        item_popularity = train_df.groupBy("item_idx") \
            .count() \
            .withColumnRenamed("count", "popularity")
        
        # Get recommendations for sample users
        sample_users = test_df.select("user_idx").distinct().sample(0.05, seed=42)
        recommendations = model.recommendForUserSubset(sample_users, 20)
        
        # Join with popularity
        rec_items = recommendations \
            .select(explode(col("recommendations.item_idx")).alias("item_idx")) \
            .join(item_popularity, "item_idx") \
            .agg(avg("popularity").alias("avg_popularity"))
        
        avg_rec_popularity = rec_items.collect()[0]["avg_popularity"]
        
        # Compare with overall average
        overall_avg_popularity = item_popularity.agg(avg("popularity")).collect()[0][0]
        
        popularity_ratio = avg_rec_popularity / overall_avg_popularity
        
        print(f"Popularity bias ratio: {popularity_ratio:.2f} " + 
              f"(>1 means bias towards popular items)")
        
        return popularity_ratio
    
    def _analyze_cold_start_performance(self, model, train_df, test_df):
        """Analyze performance on cold start users"""
        # Find users with few ratings in training
        user_counts = train_df.groupBy("user_idx").count()
        
        # Cold users (< 10 ratings)
        cold_users = user_counts.filter(col("count") < 10).select("user_idx")
        warm_users = user_counts.filter(col("count") >= 50).select("user_idx")
        
        # Filter test set
        cold_test = test_df.join(cold_users, "user_idx")
        warm_test = test_df.join(warm_users, "user_idx")
        
        # Evaluate separately
        from pyspark.ml.evaluation import RegressionEvaluator
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        if cold_test.count() > 0:
            cold_predictions = model.transform(cold_test)
            cold_rmse = evaluator.evaluate(cold_predictions)
            self.results['cold_user_rmse'] = cold_rmse
            print(f"Cold start user RMSE: {cold_rmse:.4f}")
        
        if warm_test.count() > 0:
            warm_predictions = model.transform(warm_test)
            warm_rmse = evaluator.evaluate(warm_predictions)
            self.results['warm_user_rmse'] = warm_rmse
            print(f"Warm user RMSE: {warm_rmse:.4f}")
    
    def save_results(self, path="/app/models/evaluation_results.json"):
        """Save evaluation results"""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {path}")
    
    def plot_metrics(self, save_path="/app/models/evaluation_plots.png"):
        """Create visualization of evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: RMSE and MAE
        metrics = ['val_rmse', 'test_rmse', 'val_mae', 'test_mae']
        values = [self.results.get(m, 0) for m in metrics]
        labels = ['Val RMSE', 'Test RMSE', 'Val MAE', 'Test MAE']
        
        axes[0, 0].bar(labels, values)
        axes[0, 0].set_title('Prediction Error Metrics')
        axes[0, 0].set_ylabel('Error')
        
        # Plot 2: Precision and Recall at K
        k_values = [5, 10, 20]
        precisions = [self.results.get(f'precision@{k}', 0) for k in k_values]
        recalls = [self.results.get(f'recall@{k}', 0) for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, precisions, width, label='Precision')
        axes[0, 1].bar(x + width/2, recalls, width, label='Recall')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_title('Precision and Recall at K')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(k_values)
        axes[0, 1].legend()
        
        # Plot 3: Coverage and Popularity Bias
        coverage = self.results.get('coverage@100', 0)
        popularity_bias = self.results.get('popularity_bias', 0)
        
        axes[1, 0].bar(['Coverage@100', 'Popularity Bias'], [coverage, popularity_bias])
        axes[1, 0].set_title('Coverage and Bias Metrics')
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # Plot 4: Cold vs Warm Performance
        if 'cold_user_rmse' in self.results and 'warm_user_rmse' in self.results:
            cold_rmse = self.results['cold_user_rmse']
            warm_rmse = self.results['warm_user_rmse']
            
            axes[1, 1].bar(['Cold Users', 'Warm Users'], [cold_rmse, warm_rmse])
            axes[1, 1].set_title('Cold Start Analysis')
            axes[1, 1].set_ylabel('RMSE')
        else:
            axes[1, 1].text(0.5, 0.5, 'Cold start analysis not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")

