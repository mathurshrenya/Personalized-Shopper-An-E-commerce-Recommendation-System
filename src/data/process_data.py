from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_timestamp
from delta.tables import DeltaTable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, spark_session=None):
        """Initialize the DataProcessor with a Spark session."""
        self.spark = spark_session or SparkSession.builder \
            .appName("E-commerce Recommendation System") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        logger.info("Spark session initialized")

    def read_interaction_data(self, input_path):
        """Read user interaction data from the input path."""
        try:
            df = self.spark.read.format("delta").load(input_path)
            logger.info(f"Successfully read data from {input_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise

    def preprocess_interactions(self, df):
        """Preprocess the interaction data."""
        processed_df = df \
            .withColumn("timestamp", to_timestamp(from_unixtime(col("timestamp")))) \
            .dropDuplicates(["user_id", "item_id", "timestamp"]) \
            .filter(col("rating").isNotNull())
        
        logger.info("Data preprocessing completed")
        return processed_df

    def create_delta_table(self, df, output_path, partition_by=None):
        """Create or update a Delta table with the processed data."""
        try:
            if partition_by:
                df.write.format("delta") \
                    .partitionBy(partition_by) \
                    .mode("overwrite") \
                    .save(output_path)
            else:
                df.write.format("delta") \
                    .mode("overwrite") \
                    .save(output_path)
            
            logger.info(f"Delta table created at {output_path}")
        except Exception as e:
            logger.error(f"Error creating Delta table: {str(e)}")
            raise

    def get_user_item_matrix(self, df):
        """Create a user-item interaction matrix."""
        user_item_matrix = df \
            .groupBy("user_id", "item_id") \
            .agg({"rating": "mean"}) \
            .withColumnRenamed("avg(rating)", "rating")
        
        logger.info("User-item matrix created")
        return user_item_matrix

def main():
    # Example usage
    processor = DataProcessor()
    
    # Read and process data
    input_path = "data/raw/interactions"
    output_path = "data/processed/interactions"
    
    df = processor.read_interaction_data(input_path)
    processed_df = processor.preprocess_interactions(df)
    
    # Create Delta table
    processor.create_delta_table(
        processed_df,
        output_path,
        partition_by=["user_id"]
    )
    
    # Create user-item matrix
    user_item_matrix = processor.get_user_item_matrix(processed_df)
    processor.create_delta_table(
        user_item_matrix,
        "data/processed/user_item_matrix"
    )

if __name__ == "__main__":
    main() 