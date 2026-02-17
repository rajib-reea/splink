# Oracle to Spark Data Loading with Splink Clustering

This guide provides comprehensive examples for loading data from Oracle Database into Spark and using Splink for entity resolution and deduplication.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Examples](#detailed-examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

This example demonstrates a complete workflow for:
1. Connecting to Oracle Database from PySpark
2. Loading data efficiently from Oracle tables
3. Preparing data for Splink
4. Performing entity resolution/deduplication using Splink
5. Saving results back to Oracle or file systems

Splink is a Python library for probabilistic record linkage (entity resolution) at scale. When combined with Spark and Oracle, it enables:
- **Large-scale deduplication**: Process millions of records from Oracle databases
- **Data integration**: Link records across multiple Oracle tables or databases
- **Data quality improvement**: Identify and resolve duplicate entities

## Prerequisites

### 1. Oracle JDBC Driver

Download the appropriate Oracle JDBC driver for your Oracle database version:

- **Oracle 19c and later**: [ojdbc11.jar](https://www.oracle.com/database/technologies/appdev/jdbc-downloads.html)
- **Oracle 12c/18c**: [ojdbc8.jar](https://www.oracle.com/database/technologies/appdev/jdbc-downloads.html)

Place the JAR file in a known location on your system.

### 2. Python Packages

Install required packages:

```bash
# Install Splink with Spark support
pip install 'splink[spark]'

# Or install separately
pip install splink
pip install pyspark
```

### 3. Oracle Database Access

Ensure you have:
- Oracle database hostname and port
- Service name or SID
- Valid username and password
- Network access to the Oracle database

## Quick Start

### Minimal Example

```python
from pyspark.sql import SparkSession
from pyspark import SparkConf
from splink.backends.spark import similarity_jar_location
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, SparkAPI, block_on

# 1. Configure Spark with Oracle JDBC driver
conf = SparkConf()
conf.set("spark.jars", "/path/to/ojdbc11.jar," + similarity_jar_location())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# 2. Load data from Oracle
df = spark.read.jdbc(
    url="jdbc:oracle:thin:@//host:1521/service",
    table="your_table",
    properties={
        "user": "username",
        "password": "password",
        "driver": "oracle.jdbc.driver.OracleDriver"
    }
)

# 3. Configure Splink
db_api = SparkAPI(spark_session=spark)
settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.JaroWinklerAtThresholds("first_name", [0.9]),
        cl.ExactMatch("surname"),
    ],
    blocking_rules_to_generate_predictions=[block_on("surname")]
)

# 4. Create linker and generate predictions
linker = Linker(df, settings, db_api)
linker.training.estimate_probability_two_random_records_match([block_on("surname")], recall=0.7)
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
linker.training.estimate_parameters_using_expectation_maximisation(block_on("surname"))

# 5. Generate clusters
predictions = linker.inference.predict(threshold_match_weight=-10)
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(predictions, 0.95)

# 6. Save results
clusters_df = clusters.as_spark_dataframe()
clusters_df.write.jdbc(
    url="jdbc:oracle:thin:@//host:1521/service",
    table="output_table",
    mode="overwrite",
    properties={"user": "username", "password": "password", "driver": "oracle.jdbc.driver.OracleDriver"}
)
```

## Detailed Examples

### 1. Python Script

See [`oracle_to_spark_example.py`](./oracle_to_spark_example.py) for a complete, production-ready example with:
- Multiple data loading strategies
- Comprehensive error handling
- Results analysis
- Multiple output formats

### 2. Jupyter Notebook

See [`oracle_to_spark_clustering.ipynb`](./oracle_to_spark_clustering.ipynb) for an interactive tutorial with:
- Step-by-step explanations
- Visual outputs
- Performance optimization tips
- Troubleshooting guidance

## Performance Optimization

### 1. Efficient Data Loading from Oracle

#### Use Partitioning for Large Tables

```python
# Parallel reads from Oracle using partitioning
df = spark.read.jdbc(
    url=oracle_jdbc_url,
    table="large_table",
    column="id",           # Numeric column for partitioning
    lowerBound=1,
    upperBound=10000000,
    numPartitions=20,      # Number of parallel reads
    properties=connection_properties
)
```

**Guidelines**:
- Choose a numeric column with uniform distribution
- Set `numPartitions` based on data size (typically 5-20)
- Ensure `lowerBound` and `upperBound` cover your data range

#### Filter Data at Source

```python
# Push down filtering to Oracle and convert date types
query = """
    (SELECT 
        id, 
        first_name, 
        surname, 
        TO_CHAR(date_of_birth, 'YYYY-MM-DD') as dob,
        city, 
        email
     FROM persons
     WHERE created_date >= TO_DATE('2020-01-01', 'YYYY-MM-DD')
       AND status = 'ACTIVE'
       AND first_name IS NOT NULL
       AND date_of_birth IS NOT NULL
    ) AS filtered_data
"""

df = spark.read.jdbc(url=oracle_jdbc_url, table=query, properties=connection_properties)
```

**Benefits**:
- Reduces data transfer from Oracle to Spark
- Leverages Oracle's query optimizer
- Filters out unnecessary records early
- Converts date types in Oracle (more efficient than PySpark conversion)

**Note**: When loading entire tables, you may need to convert date columns in PySpark:
```python
df = df.withColumn("dob", F.col("date_of_birth").cast("string"))
```
However, converting dates in the Oracle query using `TO_CHAR` is more efficient.

### 2. Spark Configuration for Splink

```python
conf = SparkConf()

# Memory settings (adjust based on data size)
conf.set("spark.driver.memory", "16g")
conf.set("spark.executor.memory", "8g")
conf.set("spark.executor.cores", "4")

# Parallelism (adjust based on cluster size)
conf.set("spark.default.parallelism", "200")
conf.set("spark.sql.shuffle.partitions", "200")

# Performance tuning
conf.set("spark.sql.adaptive.enabled", "true")
conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# For Splink specifically
conf.set("spark.sql.codegen.wholeStage", "false")
```

### 3. Splink Optimization

#### Choose Effective Blocking Rules

```python
# Good: Reduces comparison space significantly
blocking_rules_to_generate_predictions=[
    block_on("surname"),
    block_on("dob", "city"),
]

# Better: Multiple focused blocking rules
blocking_rules_to_generate_predictions=[
    block_on("surname"),
    block_on("first_name"),
    "l.dob = r.dob AND l.city = r.city",
]
```

#### Use Checkpointing

```python
# Break lineage to avoid recomputation
db_api = SparkAPI(
    spark_session=spark,
    break_lineage_method="checkpoint"  # or "delta_lake_table"
)

# Set checkpoint directory
spark.sparkContext.setCheckpointDir("./tmp_checkpoints")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Oracle JDBC Driver Not Found

**Error**: `java.lang.ClassNotFoundException: oracle.jdbc.driver.OracleDriver`

**Solution**:
```python
# Ensure JDBC driver is in Spark configuration
conf.set("spark.jars", "/full/path/to/ojdbc11.jar," + similarity_jar_location())

# Or add to environment variable
export SPARK_CLASSPATH=/path/to/ojdbc11.jar
```

#### 2. Out of Memory Errors

**Error**: `java.lang.OutOfMemoryError: Java heap space`

**Solutions**:
```python
# Increase driver memory
conf.set("spark.driver.memory", "32g")

# Use partitioning for large tables
df = spark.read.jdbc(..., numPartitions=20, ...)

# Reduce comparison space with tighter blocking rules
blocking_rules = [block_on("surname", "city")]
```

#### 3. Oracle Connection Timeout

**Error**: `java.sql.SQLRecoverableException: IO Error: Connection timed out`

**Solutions**:
```python
# Add connection timeout properties
connection_properties = {
    "user": username,
    "password": password,
    "driver": "oracle.jdbc.driver.OracleDriver",
    "oracle.net.CONNECT_TIMEOUT": "90000",  # 90 seconds
    "oracle.jdbc.ReadTimeout": "300000"     # 5 minutes
}

# Use connection pool for multiple reads
connection_properties["oracle.jdbc.timezoneAsRegion"] = "false"
```

#### 4. Splink Performance Issues

**Problem**: Splink processing is very slow

**Solutions**:
```python
# 1. Tighten blocking rules to reduce comparison space
blocking_rules = [
    block_on("surname"),  # Instead of no blocking
    "l.dob = r.dob",
]

# 2. Use checkpointing
db_api = SparkAPI(spark_session=spark, break_lineage_method="checkpoint")

# 3. Reduce data size
df = df.sample(fraction=0.1)  # Test with sample first

# 4. Increase Spark parallelism
conf.set("spark.default.parallelism", "400")
```

## Best Practices

### 1. Security

```python
import os

# Use environment variables for credentials
ORACLE_USER = os.environ.get("ORACLE_USER")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD")

# Or use Oracle Wallet
connection_properties = {
    "user": "/",  # External authentication
    "driver": "oracle.jdbc.driver.OracleDriver",
    "oracle.net.wallet_location": "/path/to/wallet"
}

# Enable SSL/TLS for production
oracle_jdbc_url = "jdbc:oracle:thin:@(DESCRIPTION=(ADDRESS=(PROTOCOL=TCPS)(HOST=...)(PORT=2484))...)"
```

### 2. Data Quality

```python
from pyspark.sql import functions as F

# Clean and standardize data before linking
df = (
    df
    .withColumn("first_name", F.trim(F.lower(F.col("first_name"))))
    .withColumn("surname", F.trim(F.lower(F.col("surname"))))
    .withColumn("email", F.lower(F.col("email")))
    .filter(F.col("first_name").isNotNull())
    .filter(F.col("surname").isNotNull())
)
```

### 3. Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log key metrics
logger.info(f"Records loaded from Oracle: {df.count()}")
logger.info(f"Records after filtering: {filtered_df.count()}")

# Monitor Spark UI
# Access at http://localhost:4040 while job is running

# Track cluster statistics
num_clusters = clusters_df.select("cluster_id").distinct().count()
avg_cluster_size = clusters_df.groupBy("cluster_id").count().agg(F.avg("count")).collect()[0][0]
logger.info(f"Total clusters: {num_clusters}, Average size: {avg_cluster_size}")
```

### 4. Testing Strategy

```python
# 1. Start with a small sample
sample_df = df.sample(fraction=0.01, seed=42)
linker = Linker(sample_df, settings, db_api)

# 2. Test blocking rules effectiveness
blocking_rules_stats = linker.count_num_comparisons_from_blocking_rules_for_prediction(
    [block_on("surname")]
)

# 3. Iterate and refine
# Adjust blocking rules, comparison thresholds based on results

# 4. Scale to full dataset
full_linker = Linker(df, settings, db_api)
```

### 5. Production Deployment

```python
# Use configuration file for settings
import yaml

with open("splink_config.yaml") as f:
    config = yaml.safe_load(f)

settings = SettingsCreator(**config)

# Implement error handling and retries
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_from_oracle(spark, url, table, properties):
    return spark.read.jdbc(url=url, table=table, properties=properties)

# Save intermediate results for recovery
df.write.parquet("checkpoints/input_data.parquet", mode="overwrite")
```

## Additional Resources

- [Splink Documentation](https://moj-analytical-services.github.io/splink/)
- [Splink Examples](https://moj-analytical-services.github.io/splink/demos/examples/examples_index.html)
- [PySpark JDBC Documentation](https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html)
- [Oracle JDBC Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/19/jjdbc/)

## Support

For questions or issues:
- Splink: [GitHub Issues](https://github.com/moj-analytical-services/splink/issues)
- Splink: [Discussions](https://github.com/moj-analytical-services/splink/discussions)

## License

These examples are provided under the same license as Splink (MIT License).
