import os
import json
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode, udf, expr, lit, when, count, sum as spark_sum, rank
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.window import Window
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create and configure Spark session."""
    return (SparkSession.builder
            .appName("RecommendationSystem")
            .config("spark.sql.warehouse.dir", "spark-warehouse")
            .getOrCreate())

def load_data(spark, data_path):
    """Load interaction data from Parquet files."""
    logger.info(f"Loading data from {data_path}")
    return spark.read.parquet(data_path)

def load_mappings(mapping_path):
    """Load user and item mappings from JSON files."""
    logger.info(f"Loading mappings from {mapping_path}")
    with open(mapping_path, 'r') as f:
        return json.load(f)

def preprocess_data(df, user_map, item_map):
    """Preprocess the interaction data."""
    logger.info("Preprocessing data")
    # Create UDFs for mapping string IDs to integer IDs
    user_map_udf = udf(lambda x: user_map.get(x), IntegerType())
    item_map_udf = udf(lambda x: item_map.get(x), IntegerType())
    
    # Convert string IDs to integer IDs using the mappings
    df = df.withColumn("user_id_int", user_map_udf(col("user_id")))
    df = df.withColumn("item_id_int", item_map_udf(col("item_id")))
    
    # Drop rows with null IDs (if any)
    df = df.dropna(subset=["user_id_int", "item_id_int"])
    
    # Drop the original string ID columns
    df = df.drop("user_id", "item_id")
    
    # Rename the integer ID columns
    df = df.withColumnRenamed("user_id_int", "user_id")
    df = df.withColumnRenamed("item_id_int", "item_id")
    
    return df

def train_model(df, max_iter=10, reg_param=0.1, rank=10):
    """Train ALS model on the interaction data."""
    logger.info("Training ALS model")
    als = ALS(
        maxIter=max_iter,
        regParam=reg_param,
        rank=rank,
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    
    model = als.fit(df)
    return model

def generate_recommendations(model, test_df, user_map, item_map, k=10):
    """Generate top-k recommendations for each user."""
    # Get unique users from test set
    users = test_df.select("user_id").distinct()
    
    # Generate recommendations
    recommendations = model.recommendForUserSubset(users, k)
    
    # Explode recommendations to get individual items
    exploded_recs = recommendations.select(
        "user_id",
        expr("explode(recommendations)").alias("rec")
    ).select(
        "user_id",
        col("rec.item_id").alias("item_id"),
        col("rec.rating").alias("predicted_rating")
    )
    
    # Convert numeric IDs back to original IDs
    reverse_user_map = {v: k for k, v in user_map.items()}
    reverse_item_map = {v: k for k, v in item_map.items()}
    
    # Create UDFs for ID conversion
    convert_user_id = udf(lambda x: reverse_user_map.get(x, str(x)), StringType())
    convert_item_id = udf(lambda x: reverse_item_map.get(x, str(x)), StringType())
    
    # Apply ID conversions
    final_recommendations = exploded_recs \
        .withColumn("user_id", convert_user_id(col("user_id"))) \
        .withColumn("item_id", convert_item_id(col("item_id")))
    
    return final_recommendations

def calculate_metrics(model, test_df, k=10):
    """Calculate various evaluation metrics for the recommendation model."""
    # Generate recommendations for test users
    user_recs = model.recommendForUserSubset(test_df.select("user_id").distinct(), k)
    
    # Explode recommendations to get individual items
    recs_exploded = user_recs.select(
        "user_id",
        expr("explode(recommendations)").alias("rec")
    ).select(
        "user_id",
        col("rec.item_id").alias("item_id"),
        col("rec.rating").alias("predicted_rating")
    )
    
    # Calculate RMSE and MAE
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse", 
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)
    
    # Calculate Precision@K and Recall@K
    # Create window spec for ranking
    window_spec = Window.partitionBy("user_id").orderBy(col("predicted_rating").desc())
    
    # Rank recommendations
    ranked_recs = recs_exploded.withColumn("rec_rank", rank().over(window_spec))
    
    # Get top K recommendations
    top_k_recs = ranked_recs.filter(col("rec_rank") <= k)
    
    # Join with test data to find relevant items
    relevant_items = test_df.filter(col("rating") >= 4.0)  # Items rated 4 or higher are considered relevant
    
    # Calculate precision and recall
    hits = top_k_recs.join(
        relevant_items,
        ["user_id", "item_id"],
        "inner"
    ).count()
    
    total_recs = top_k_recs.count()
    total_relevant = relevant_items.count()
    
    precision = hits / total_recs if total_recs > 0 else 0
    recall = hits / total_relevant if total_relevant > 0 else 0
    
    # Calculate NDCG@K
    # Create window spec for test data ranking
    test_window_spec = Window.partitionBy("user_id").orderBy(col("rating").desc())
    
    # Rank test data
    ranked_test = test_df.withColumn("test_rank", rank().over(test_window_spec))
    
    # Calculate DCG
    dcg = top_k_recs.join(
        ranked_test,
        ["user_id", "item_id"],
        "left"
    ).withColumn(
        "dcg",
        expr("CASE WHEN rating IS NOT NULL THEN 1.0 / log2(rec_rank + 1) ELSE 0 END")
    ).agg({"dcg": "sum"}).collect()[0][0]
    
    # Calculate IDCG
    idcg = ranked_test.filter(col("test_rank") <= k).withColumn(
        "idcg",
        expr("1.0 / log2(test_rank + 1)")
    ).agg({"idcg": "sum"}).collect()[0][0]
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return {
        "rmse": rmse,
        "mae": mae,
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg
    }

def save_recommendations(recommendations, output_path):
    """Save recommendations to Parquet format."""
    logger.info(f"Saving recommendations to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    recommendations.write.parquet(output_path, mode="overwrite")

def main():
    """Main pipeline execution."""
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("RecommendationSystem") \
            .getOrCreate()
        
        # Load and preprocess data
        logger.info("Loading data from data/raw/interactions")
        interactions_df = spark.read.parquet("data/raw/interactions")
        
        # Load mappings
        logger.info("Loading mappings from data/processed/user_map.json")
        with open("data/processed/user_map.json", "r") as f:
            user_map = json.load(f)
        
        logger.info("Loading mappings from data/processed/item_map.json")
        with open("data/processed/item_map.json", "r") as f:
            item_map = json.load(f)
        
        # Preprocess data
        logger.info("Preprocessing data")
        processed_df = preprocess_data(interactions_df, user_map, item_map)
        
        # Split data into train and test sets
        train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)
        
        # Train model
        logger.info("Training ALS model")
        model = train_model(train_df)
        
        # Generate recommendations
        logger.info("Generating 10 recommendations per user")
        recommendations = generate_recommendations(model, test_df, user_map, item_map)
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        metrics = calculate_metrics(model, test_df)
        
        # Save metrics to file
        metrics_dir = "data/processed/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(metrics_dir, f"metrics_{timestamp}.json")
        
        # Add timestamp to metrics
        metrics["timestamp"] = timestamp
        metrics["model_version"] = "1.0"
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Log metrics
        logger.info("Evaluation Metrics:")
        for metric, value in metrics.items():
            if metric not in ["timestamp", "model_version"]:
                logger.info(f"{metric}: {value:.4f}")
        
        # Save recommendations
        logger.info("Saving recommendations to data/processed/recommendations")
        recommendations.write.parquet("data/processed/recommendations", mode="overwrite")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 