---
tags:
  - Clustering
  - Blocking
  - Scoring
---

# Creating Clusters in Splink

This guide provides a comprehensive overview of how to create clusters in Splink, including information on blocking rules, scoring, and the clustering process.

## Overview

Clustering is the final step in the Splink record linkage pipeline. After comparing record pairs and calculating match probabilities, clustering groups records together that represent the same real-world entity. The result is a cluster identifier that can be used to identify groups of duplicate or linked records.

The Splink clustering process involves three main stages:

1. **Blocking**: Reducing the number of record pairs to compare
2. **Scoring**: Calculating match probabilities for record pairs
3. **Clustering**: Grouping records into clusters based on match probabilities

## The Clustering Workflow

Here's a typical workflow for creating clusters in Splink:

```python
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

# Initialize the database API
db_api = DuckDBAPI()

# Load your data
df = splink_datasets.fake_1000

# Define your Splink settings
settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=[
        cl.JaroWinklerAtThresholds("first_name", [0.9, 0.7]),
        cl.JaroAtThresholds("surname", [0.9, 0.7]),
        cl.DateOfBirthComparison(
            "dob",
            input_is_string=True,
            datetime_metrics=["year", "month"],
            datetime_thresholds=[1, 1],
        ),
        cl.ExactMatch("city").configure(term_frequency_adjustments=True),
        cl.EmailComparison("email"),
    ],
    blocking_rules_to_generate_predictions=[
        block_on("first_name"),
        block_on("surname"),
    ]
)

# Create the linker
linker = Linker(df, settings, db_api)

# Train the model (estimate parameters)
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "surname")],
    recall=0.7,
)

linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")
)

linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))

# Generate predictions
pairwise_predictions = linker.inference.predict(threshold_match_weight=-10)

# Create clusters at a specific threshold
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions, threshold_match_probability=0.95
)

# Get results as a dataframe
df_clusters = clusters.as_pandas_dataframe()
```

## Understanding Blocking

Blocking is a critical technique for making record linkage computationally feasible. Without blocking, comparing all possible pairs of records would require $\frac{n(n-1)}{2}$ comparisons, which becomes infeasible for large datasets.

### What are Blocking Rules?

Blocking rules define which record pairs should be compared. A blocking rule specifies conditions under which two records should be brought together for comparison. For example:

```python
block_on("first_name")
```

This rule means "only compare records where the first name is exactly the same."

### Multiple Blocking Rules

Splink allows you to specify multiple blocking rules. Record pairs that satisfy **any** of your blocking rules will be compared:

```python
blocking_rules_to_generate_predictions=[
    block_on("first_name"),
    block_on("surname"),
    block_on("dob"),
]
```

With these rules, a record pair will be compared if:
- The first names match exactly, OR
- The surnames match exactly, OR
- The dates of birth match exactly

### Best Practices for Blocking

1. **Use multiple strict rules rather than one loose rule**: This helps capture more true matches while keeping the number of comparisons manageable.

2. **Consider data quality**: If a field has many errors or missing values, blocking on it alone may miss true matches.

3. **Balance recall and computational cost**: Looser blocking rules capture more matches but increase computation time.

4. **Test your blocking rules**: Use Splink's blocking analysis tools to understand how many comparisons each rule generates:

```python
linker.exploratory.count_num_comparisons_from_blocking_rule(
    block_on("first_name")
)
```

### Common Blocking Strategies

- **Exact match on stable identifiers**: e.g., `block_on("email")`, `block_on("phone_number")`
- **Combination blocking**: e.g., `block_on("first_name", "dob")`
- **Phonetic blocking**: Block on phonetic versions of names to catch spelling variations
- **First N characters**: Block on the first few characters of a field

For more detailed information on blocking, see the [Blocking Rules topic guide](../blocking/blocking_rules.md).

## Understanding Scoring (Match Weights and Probabilities)

After blocking determines which record pairs to compare, Splink scores each pair to determine how likely they are to be a match.

### Match Weights

Splink uses the Fellegi-Sunter model to calculate a **match weight** for each record pair. The match weight is a log-likelihood ratio that represents the evidence for or against a match:

- **Positive match weights**: Evidence that the records match
- **Negative match weights**: Evidence that the records don't match
- **Zero match weight**: Neutral evidence

Match weights are additive - the total match weight is the sum of match weights from all comparisons.

### Match Probabilities

Match probabilities are derived from match weights using the formula:

$$
P(\text{match}) = \frac{e^{\text{match weight}}}{1 + e^{\text{match weight}}}
$$

Match probabilities range from 0 to 1:
- **Close to 1**: Strong evidence of a match
- **Close to 0**: Strong evidence of a non-match
- **Around 0.5**: Uncertain

### Comparison Functions

Splink compares records using **comparison functions** that you define in your settings. Each comparison function:

1. Compares a specific field (or fields) between two records
2. Outputs a comparison level based on the degree of similarity
3. Contributes to the overall match weight

Example comparison:

```python
cl.JaroWinklerAtThresholds("first_name", [0.9, 0.7])
```

This creates a comparison on `first_name` with multiple levels:
- Exact match (Jaro-Winkler = 1.0)
- High similarity (Jaro-Winkler >= 0.9)
- Medium similarity (Jaro-Winkler >= 0.7)
- Low similarity (Jaro-Winkler < 0.7)

### Model Training

Before scoring, Splink needs to estimate the parameters of the Fellegi-Sunter model (the $m$ and $u$ probabilities). This is done through training:

```python
# Estimate the probability that two random records match
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "surname")],
    recall=0.7,
)

# Estimate u probabilities using random sampling
linker.training.estimate_u_using_random_sampling(max_pairs=1e6)

# Estimate m probabilities using Expectation-Maximisation
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("first_name", "surname")
)
```

For more information on the statistical model, see the [Fellegi-Sunter Model topic guide](../theory/fellegi_sunter.md).

## The Clustering Process

Once you have scored record pairs, clustering groups records together based on their pairwise match probabilities.

### Connected Components Algorithm

Splink uses the **connected components** algorithm for clustering. This algorithm treats your data as a graph:

- **Nodes**: Individual records
- **Edges**: Pairwise predictions above your threshold

Records are clustered together if there is a path connecting them through edges above the threshold, even if they are not directly linked.

#### Example

Consider three records: A, B, and C.

- Record A matches B with probability 0.98
- Record B matches C with probability 0.96
- Record A matches C with probability 0.85

At a threshold of 0.95, A and B are linked, and B and C are linked. The connected components algorithm will cluster all three records together, even though A and C are below the threshold.

### Creating Clusters

Use the `cluster_pairwise_predictions_at_threshold` method to create clusters:

```python
clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
    pairwise_predictions, 
    threshold_match_probability=0.95
)
```

You can specify the threshold in two ways:

1. **Match probability**: `threshold_match_probability=0.95` (records with probability >= 0.95 are matches)
2. **Match weight**: `threshold_match_weight=3.0` (records with weight >= 3.0 are matches)

### Choosing a Threshold

The threshold determines which links are strong enough to be considered matches. Choosing the right threshold is crucial:

- **Higher threshold (e.g., 0.99)**: Fewer false positives, but may miss some true matches
- **Lower threshold (e.g., 0.70)**: Captures more true matches, but more false positives

You can use Splink's evaluation tools to help choose an appropriate threshold:

```python
# If you have labelled data
linker.evaluation.accuracy_analysis_from_labels_table("labels_table")

# Use the threshold selection tool
from splink.exploratory import threshold_selection_tool_from_labels_table
threshold_selection_tool_from_labels_table(linker, "labels_table")
```

### Working with Clusters

Once you have created clusters, you can:

1. **Convert to pandas DataFrame**:
```python
df_clusters = clusters.as_pandas_dataframe()
```

2. **Visualize clusters**:
```python
linker.visualisations.cluster_studio_dashboard(
    df_predict=pairwise_predictions,
    cluster_ids=[1, 2, 3],  # Specify which clusters to visualize
    cluster_name="cluster_id",
)
```

3. **Compute graph metrics**:
```python
graph_metrics = linker.clustering.compute_graph_metrics(
    pairwise_predictions,
    threshold_match_probability=0.95
)
```

For more information on evaluating clusters, see the [Cluster Evaluation topic guide](../evaluation/clusters/overview.md).

## Advanced Clustering Topics

### One-to-One Clustering

In some scenarios, you may want to enforce that each record can match to at most one other record (e.g., linking two datasets where each record in dataset A should match to at most one record in dataset B).

Splink provides one-to-one clustering functionality:

```python
from splink import one_to_one_clustering

clusters = one_to_one_clustering(
    pairwise_predictions,
    threshold_match_probability=0.95
)
```

### Cluster Metrics

Splink can compute various metrics on clusters to help evaluate quality:

- **Cluster size**: Number of nodes in each cluster
- **Cluster density**: Proportion of possible edges that exist in a cluster
- **Node degree**: Number of connections each node has

```python
from splink.internals.graph_metrics import GraphMetricsResults

metrics = linker.clustering.compute_graph_metrics(
    pairwise_predictions,
    threshold_match_probability=0.95
)

# Access metrics
df_metrics = metrics.as_pandas_dataframe()
```

For more information, see the [Graph Metrics topic guide](../evaluation/clusters/graph_metrics.md).

## Common Issues and Solutions

### Large Clusters

If you see unexpectedly large clusters, this may indicate:

- **Threshold too low**: Lower the threshold to reduce false positive links
- **Blocking rules too loose**: Tighten blocking rules to reduce spurious comparisons
- **Data quality issues**: Clean your data to reduce false matches

Use the Cluster Studio dashboard to investigate large clusters:

```python
linker.visualisations.cluster_studio_dashboard(
    df_predict=pairwise_predictions,
    cluster_ids=[large_cluster_id],
)
```

### Missing Clusters

If expected clusters are not formed, this may indicate:

- **Threshold too high**: Increase the threshold to capture more matches
- **Blocking rules too strict**: Add more blocking rules to ensure potential matches are compared
- **Model parameters not well-estimated**: Retrain your model or use different training data

### Performance Considerations

For large datasets, clustering can be computationally expensive. Tips for improving performance:

1. **Use tighter blocking rules** to reduce the number of comparisons
2. **Use appropriate thresholds** when calling `predict()` to filter out unlikely matches early
3. **Use a scalable backend** like Spark or Athena for very large datasets
4. **Consider salting** for extremely skewed data (see [Performance topic guide](../performance/drivers_of_performance.md))

## Summary

Creating clusters in Splink involves:

1. **Defining blocking rules** to determine which record pairs to compare
2. **Training the model** to estimate match probabilities
3. **Scoring record pairs** using comparisons and the Fellegi-Sunter model
4. **Clustering** records based on match probabilities and a threshold
5. **Evaluating** cluster quality using visualizations and metrics

By understanding each of these stages, you can build effective record linkage models and generate high-quality clusters for your data.

## Further Reading

- [Blocking Rules topic guide](../blocking/blocking_rules.md)
- [Fellegi-Sunter Model topic guide](../theory/fellegi_sunter.md)
- [Linked Data as Graphs topic guide](../theory/linked_data_as_graphs.md)
- [Cluster Evaluation topic guide](../evaluation/clusters/overview.md)
- [Graph Metrics topic guide](../evaluation/clusters/graph_metrics.md)
- [Performance Optimization topic guide](../performance/drivers_of_performance.md)
- [API Documentation for Clustering](../../api_docs/linker_clustering.md)
