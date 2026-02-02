"""
Example: Loading Data from Oracle into Spark and Using Splink for Clustering

This example demonstrates how to:
1. Connect to an Oracle database using PySpark
2. Load data from Oracle into a Spark DataFrame
3. Use Splink to perform entity resolution/clustering on the data

Prerequisites:
- Oracle JDBC driver (ojdbc8.jar or ojdbc11.jar)
- PySpark installed: pip install 'splink[spark]'
- Splink installed: pip install splink
- Access to an Oracle database
"""

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# ============================================================================
# Step 1: Configure Spark with Oracle JDBC Driver
# ============================================================================

# You need to download the Oracle JDBC driver from:
# https://www.oracle.com/database/technologies/appdev/jdbc-downloads.html
# 
# For Oracle 19c and later: ojdbc11.jar
# For Oracle 12c/18c: ojdbc8.jar

# Path to your Oracle JDBC driver
ORACLE_JDBC_JAR_PATH = "/path/to/ojdbc11.jar"

# Oracle database connection details
ORACLE_HOST = "your-oracle-host"
ORACLE_PORT = "1521"
ORACLE_SERVICE_NAME = "your-service-name"  # or use SID
ORACLE_USER = "your-username"
ORACLE_PASSWORD = "your-password"

# Configure Spark
conf = SparkConf()
conf.set("spark.driver.memory", "12g")
conf.set("spark.default.parallelism", "8")
conf.set("spark.sql.codegen.wholeStage", "false")

# Add Oracle JDBC driver and Splink's similarity JAR
from splink.backends.spark import similarity_jar_location

similarity_jar_path = similarity_jar_location()
conf.set("spark.jars", f"{ORACLE_JDBC_JAR_PATH},{similarity_jar_path}")

# Create Spark session
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession(sc)
spark.sparkContext.setCheckpointDir("./tmp_checkpoints")

print("Spark session created successfully!")

# ============================================================================
# Step 2: Load Data from Oracle Database
# ============================================================================

# Construct Oracle JDBC URL
oracle_jdbc_url = (
    f"jdbc:oracle:thin:@//{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE_NAME}"
)

# Connection properties
connection_properties = {
    "user": ORACLE_USER,
    "password": ORACLE_PASSWORD,
    "driver": "oracle.jdbc.driver.OracleDriver"
}

# Method 1: Load entire table
# Simplest method, suitable for small to medium tables
table_name = "YOUR_TABLE_NAME"
df = spark.read.jdbc(
    url=oracle_jdbc_url,
    table=table_name,
    properties=connection_properties
)

# Note: When loading full table, you may need to convert date types later
# in PySpark using: df.withColumn("dob", F.col("date_of_birth").cast("string"))

# Method 2: Load with SQL query (recommended for large tables)
# This allows you to filter data at the source
# Convert dates to strings in Oracle for better compatibility
query = """
    (SELECT 
        person_id,
        first_name,
        surname,
        TO_CHAR(date_of_birth, 'YYYY-MM-DD') as dob,
        city,
        email
     FROM your_table_name
     WHERE active_flag = 'Y'
       AND date_of_birth IS NOT NULL
    ) AS filtered_data
"""

df = spark.read.jdbc(
    url=oracle_jdbc_url,
    table=query,
    properties=connection_properties
)

# Method 3: Load with partitioning (recommended for very large tables)
# This enables parallel reads from Oracle
# Note: When using table name directly, you may need to convert date columns
# in PySpark. For better performance, use a query with TO_CHAR instead.
df = spark.read.jdbc(
    url=oracle_jdbc_url,
    table=table_name,
    column="person_id",  # Column to partition on (should be numeric)
    lowerBound=1,
    upperBound=1000000,
    numPartitions=10,  # Number of parallel connections
    properties=connection_properties
)

# If date columns weren't converted in Oracle query, convert them now:
# df = df.withColumn("dob", F.col("date_of_birth").cast("string"))

# Display sample data
print("\n=== Sample Data from Oracle ===")
df.show(5)
df.printSchema()

# ============================================================================
# Step 3: Data Preparation for Splink
# ============================================================================

# Ensure data types are compatible with Splink
from pyspark.sql import functions as F

# Note: If date wasn't converted to string in Oracle query, convert it here:
# df = df.withColumn("dob", F.col("dob").cast("string"))
# Or use TO_CHAR in Oracle query (recommended for better performance)

# Handle NULL values if needed
# df = df.fillna({"email": "", "city": ""})

# Add a unique identifier if not present
# df = df.withColumn("unique_id", F.monotonically_increasing_id())

print("\n=== Data prepared for Splink ===")
df.show(5)

# ============================================================================
# Step 4: Configure Splink for Deduplication/Clustering
# ============================================================================

import splink.comparison_library as cl
from splink import Linker, SettingsCreator, SparkAPI, block_on

# Create SparkAPI instance
db_api = SparkAPI(
    spark_session=spark,
    break_lineage_method="checkpoint"  # or "delta_lake_table"
)

# Define comparison settings
settings = SettingsCreator(
    link_type="dedupe_only",  # Use "dedupe_only" for deduplication within one dataset
    comparisons=[
        # Compare first names with Jaro-Winkler similarity
        cl.JaroWinklerAtThresholds("first_name", [0.9, 0.7]),
        
        # Compare surnames with Jaro similarity
        cl.JaroAtThresholds("surname", [0.9, 0.7]),
        
        # Compare dates of birth
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
            datetime_metrics=["year", "month"],
            datetime_thresholds=[1, 1],
        ),
        
        # Compare city with exact match and term frequency adjustments
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        
        # Compare email addresses
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        # Only compare records that share the same first name OR surname
        block_on("first_name"),
        block_on("surname"),
    ],
    retain_matching_columns=True,
    retain_intermediate_calculation_columns=True,
)

# Create Linker
linker = Linker(df, settings, db_api)

print("\n=== Splink Linker created successfully ===")

# ============================================================================
# Step 5: Train the Model
# ============================================================================

print("\n=== Training Splink model ===")

# Estimate probability that two random records match
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "surname")],
    recall=0.7,
)

# Estimate u probabilities using random sampling
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Estimate m probabilities using expectation maximisation
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")
)

linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("dob")
)

print("\n=== Model training complete ===")

# ============================================================================
# Step 6: Generate Predictions and Clusters
# ============================================================================

print("\n=== Generating predictions ===")

# Predict pairwise links
pairwise_predictions = linker.inference.predict(threshold_match_weight=-10)

# View some predictions
print("\n=== Sample Pairwise Predictions ===")
pairwise_predictions_df = pairwise_predictions.as_spark_dataframe()
pairwise_predictions_df.show(10)

# Cluster predictions to create groups of matching records
print("\n=== Clustering records ===")
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions, 
    threshold_match_probability=0.95
)

# Convert to Spark DataFrame
clusters_df = clusters.as_spark_dataframe()

print("\n=== Sample Clusters ===")
clusters_df.show(20)

# ============================================================================
# Step 7: Save Results
# ============================================================================

# Save back to Oracle (if you have write permissions)
output_table = "DEDUPLICATED_RECORDS"

clusters_df.write.jdbc(
    url=oracle_jdbc_url,
    table=output_table,
    mode="overwrite",  # or "append"
    properties=connection_properties
)

print(f"\n=== Results saved to Oracle table: {output_table} ===")

# Alternatively, save to file formats
# CSV
clusters_df.coalesce(1).write.csv(
    "output/clusters.csv",
    mode="overwrite",
    header=True
)

# Parquet (more efficient for large datasets)
clusters_df.write.parquet(
    "output/clusters.parquet",
    mode="overwrite"
)

print("\n=== Results also saved to local files ===")

# ============================================================================
# Step 8: Analysis of Results
# ============================================================================

# Count number of clusters
num_clusters = clusters_df.select("cluster_id").distinct().count()
print(f"\nTotal number of clusters: {num_clusters}")

# Count cluster sizes
cluster_sizes = clusters_df.groupBy("cluster_id").count()
cluster_sizes.orderBy(F.desc("count")).show(10)

# Show example of a cluster with multiple records
sample_cluster = (
    clusters_df
    .groupBy("cluster_id")
    .count()
    .filter(F.col("count") > 1)
    .limit(1)
    .select("cluster_id")
    .collect()
)

if sample_cluster:
    cluster_id_value = sample_cluster[0]["cluster_id"]
    print(f"\n=== Example cluster with ID: {cluster_id_value} ===")
    clusters_df.filter(F.col("cluster_id") == cluster_id_value).show(truncate=False)

# Stop Spark session
spark.stop()
print("\n=== Processing complete ===")
