# Connected Component Computation in MapReduce: Implementation and Experimental Analysis of the CCF Algorithm

---

## 1. Description of the Adopted Solution

### 1.1 Problem Statement

Finding connected components in a graph is a fundamental problem in graph theory with applications in social network analysis, record linkage, image segmentation, and data mining [1]. Given an undirected graph *G = (V, E)*, the objective is to partition *V* into disjoint sets *C = {C₁, C₂, ..., Cₙ}* such that for each component *Cᵢ*, there exists a path between any two vertices *vₖ, vₗ ∈ Cᵢ*, and no path exists between vertices in distinct components.

As graph sizes have grown to billions of nodes and edges — driven by social networks, web graphs, and entity resolution systems — sequential algorithms are no longer feasible. Distributed computing frameworks such as MapReduce and Apache Spark provide a means to scale graph algorithms across clusters of commodity machines.

### 1.2 Adopted Approach

We adopt the **Connected Component Finder (CCF)** algorithm proposed by Kardes et al. [1], which identifies connected components through iterative MapReduce jobs. The key idea is to represent each component by the smallest node ID it contains. Through repeated iterations of two MapReduce jobs — **CCF-Iterate** and **CCF-Dedup** — minimum node IDs are propagated along edges until every node in a component maps to the same minimum ID.

The algorithm takes as input an edge list and produces a mapping from each node to its component identifier (the smallest node ID in its component). The iterative process terminates when no new pairs are generated in a given iteration, as tracked by a global counter.

### 1.3 Implementation Framework

We implement the CCF algorithm using **Apache Spark** (PySpark 4.0.1 and Scala), leveraging Spark's Resilient Distributed Datasets (RDDs) as the distributed data abstraction. RDDs provide:

- **Immutability and fault tolerance**: Each RDD transformation creates a new RDD; lineage enables automatic recovery from node failures.
- **Lazy evaluation**: Transformations are not computed until an action (e.g., `count()`, `collect()`) triggers execution, enabling Spark's optimizer to plan the computation graph.
- **In-memory caching**: Intermediate RDDs can be persisted in memory across iterations using `.cache()`, avoiding repeated recomputation from disk.

The MapReduce paradigm maps naturally to Spark's RDD API: the *map phase* corresponds to `flatMap` transformations, and the *reduce phase* corresponds to `groupByKey` followed by per-key aggregation. The iterative structure of CCF is implemented as a `while` loop that repeatedly applies CCF-Iterate and CCF-Dedup until convergence.

### 1.4 Relation to the Original Paper

The original CCF paper [1] was implemented in Java using Hadoop MapReduce and demonstrated on a graph with ~6 billion nodes and ~92 billion edges on an 80-node Hadoop cluster. Our implementation preserves the algorithmic logic faithfully while adapting to the Spark RDD API. Key differences from the original Hadoop implementation include:

- **RDD-based data flow** instead of HDFS file-based intermediate storage between iterations.
- **In-memory caching** of intermediate RDDs between iterations, reducing I/O overhead.
- **The NewPair counter** is implemented via manual counting in the reduce function (Python) or Spark `LongAccumulator` (Scala), rather than Hadoop's global counter mechanism.
- **CCF-Dedup** uses Spark's `distinct()` (Python) or `reduceByKey` on composite keys (Scala), which is semantically equivalent to the paper's composite-key deduplication pattern.

One notable implementation detail: the **secondary sort variant** (Section 2.2) in the paper uses Hadoop's secondary sort mechanism with custom partitioning to deliver values to the reducer in sorted order *without loading them into memory*. In our Spark implementation, we approximate this with `groupByKey().mapValues(sorted(...))`, which does collect values into memory before sorting. This means our secondary sort variant does not achieve the O(1) memory guarantee of the paper's Hadoop implementation at the reducer level. A true Spark equivalent would require `repartitionAndSortWithinPartitions` with a custom partitioner — a worthwhile optimisation for production use on graphs with very large components.

---

## 2. Algorithms

### 2.1 Overview

The CCF module consists of two MapReduce jobs executed iteratively:

1. **CCF-Iterate**: Propagates minimum node IDs along edges.
2. **CCF-Dedup**: Removes duplicate pairs between iterations.

The process repeats until no new pairs are generated (tracked by a global counter `NewPair`). The output is a set of *(node, componentID)* pairs where `componentID` is the smallest node ID in the component.

### 2.2 CCF-Iterate — Basic Version (Figure 2 of [1])

The basic CCF-Iterate constructs adjacency lists and propagates minimum IDs.

**Map Phase:**
For each input pair *(key, value)*, emit both *(key, value)* and *(value, key)*. This ensures each node appears as a key with all its neighbours as values, effectively constructing a bidirectional adjacency list.

**Reduce Phase:**
For each key (node) with its grouped values (neighbours):
1. Find the minimum value `min` across all values and the key itself.
2. If `min < key` (i.e., a smaller ID exists in the neighbourhood):
   - Emit *(key, min)* — the key now maps to the smaller ID.
   - For each value ≠ min, emit *(value, min)* — propagate the minimum to all neighbours.
   - Increment the `NewPair` counter for each such emission.
3. If `min ≥ key`, emit nothing (this node is already the local minimum).

**Space complexity:** O(N) per reducer, where N is the size of the adjacency list, since all values must be stored in a list to iterate twice (once to find the minimum, once to emit pairs).

**RDD Translation:**
```python
mapped = pairs_rdd.flatMap(lambda pair: [(pair[0], pair[1]), (pair[1], pair[0])])
grouped = mapped.groupByKey()
# reduce_fn iterates values twice: find min, then emit pairs
```

### 2.3 CCF-Iterate — Secondary Sort Version (Figure 3 of [1])

The secondary sort variant improves memory efficiency by delivering values to the reducer in sorted order, so the first value is guaranteed to be the minimum.

**Map Phase:** Identical to the basic version.

**Reduce Phase:**
1. The first value from the sorted iterator is `minValue`.
2. If `minValue < key`:
   - Emit *(key, minValue)*.
   - For each remaining value in the iterator, emit *(value, minValue)* and increment `NewPair`.
3. If `minValue ≥ key`, emit nothing.

**Space complexity:** O(1) per reducer in the original Hadoop implementation (values are streamed from disk in sorted order). In our Spark implementation, this is approximated with an explicit sort after `groupByKey`, which still requires O(N) memory — a known limitation of the RDD-based approach (see Section 1.4).

**RDD Translation:**
```python
grouped = mapped.groupByKey().mapValues(lambda vals: sorted(vals))
# reduce_fn: first element of sorted list is minValue; single pass
```

### 2.4 CCF-Dedup (Figure 4 of [1])

CCF-Iterate may emit the same pair multiple times within a single iteration. CCF-Dedup removes these duplicates to reduce the input size for the next iteration.

**Map Phase:** For each pair *(key, value)*, create a composite key `temp = (key, value)` and emit *(temp, null)*.

**Reduce Phase:** For each unique composite key, emit *(key.entity1, key.entity2)*.

This is effectively a deduplication operation. In Spark, it maps directly to `distinct()` or equivalently `map(pair => (pair, null)).reduceByKey(...).map(_._1)`.

**RDD Translation:**
```python
# Python: simple and idiomatic
return pairs_rdd.distinct()

# Scala: explicit composite-key pattern matching the paper
pairsRDD.map(pair => (pair, null)).reduceByKey((a, _) => a).map(_._1)
```

### 2.5 Full CCF Pipeline

The complete algorithm is:

```
Input: Edge list E
Output: (node, componentID) mapping

pairs ← E
repeat:
    pairs ← CCF-Iterate(pairs)      // Propagate minimum IDs
    pairs ← CCF-Dedup(pairs)        // Remove duplicates
until NewPair counter = 0
return pairs
```

**Convergence:** The algorithm converges when no new pairs are generated in an iteration. At each iteration, minimum IDs propagate at least one hop further. For a graph with diameter *d*, convergence is guaranteed in O(log d) iterations due to the doubling nature of the propagation: each node's minimum ID reach approximately doubles per iteration.

### 2.6 Correctness

The algorithm correctly identifies connected components because:

1. **Monotonicity**: Component IDs can only decrease (each node maps to the minimum in its neighbourhood), ensuring convergence.
2. **Propagation**: The map phase emits edges in both directions, ensuring bidirectional reachability.
3. **Completeness**: The iteration continues until no new pairs are generated, meaning every node has converged to the global minimum ID of its component.

---

## 3. Experimental Analysis

### 3.1 Experimental Setup

All experiments were conducted on a single machine (Apple M-series, 8 cores) using PySpark 4.0.1 in local mode (`local[*]`). While this does not replicate a true distributed cluster, it enables controlled comparison of algorithmic behaviour (iteration counts, convergence patterns) across graph topologies.

Both algorithm variants — **Basic** (Figure 2) and **SecondarySort** (Figure 3) — were evaluated on three classes of synthetic graphs:

1. **Random graphs** (Erdős–Rényi model): Increasing size from 50 to 5,000 nodes with edge density ≈ 3× node count.
2. **Chain graphs**: Linear paths of 10 to 500 nodes representing worst-case diameter.
3. **Cluster graphs**: Multiple dense clusters with optional inter-cluster edges to control component count.

### 3.2 Experiment 1: Random Graphs of Increasing Size

| Nodes | Edges  | Algorithm     | Iterations | Time (s) | Components |
|------:|-------:|---------------|:----------:|---------:|-----------:|
|    50 |    100 | Basic         |     5      |    5.90  |     1      |
|    50 |    100 | SecondarySort |     5      |    5.06  |     1      |
|   100 |    300 | Basic         |     5      |    5.09  |     1      |
|   100 |    300 | SecondarySort |     5      |    5.23  |     1      |
|   500 |  1,500 | Basic         |     6      |    5.29  |     1      |
|   500 |  1,500 | SecondarySort |     6      |    5.67  |     1      |
| 1,000 |  3,000 | Basic         |     6      |    6.89  |     1      |
| 1,000 |  3,000 | SecondarySort |     6      |    6.26  |     1      |
| 2,000 |  6,000 | Basic         |     6      |    6.67  |     1      |
| 2,000 |  6,000 | SecondarySort |     6      |    5.76  |     1      |
| 5,000 | 15,000 | Basic         |     6      |    6.58  |     1      |
| 5,000 | 15,000 | SecondarySort |     6      |    6.78  |     1      |

**Key findings:**

- **Iteration count is nearly constant (5–6)** regardless of graph size (50 to 5,000 nodes). This is consistent with the paper's observation that real-world graphs have small effective diameters [1, Section IV], and with well-known small-world properties of random graphs [2].
- **Runtime increases modestly** (5.0s → 6.8s for a 100× increase in graph size), indicating that at this scale the fixed overhead of Spark job initialization dominates over per-iteration data processing.
- **Both variants yield identical iteration counts**, confirming that they implement the same convergence logic and differ only in reducer memory management.
- At edge density ≈ 3× node count, all graphs form a single connected component, consistent with the Erdős–Rényi connectivity threshold.

### 3.3 Experiment 2: Chain Graphs (Worst-Case Diameter)

| Nodes | Edges | Algorithm     | Iterations | Time (s) |
|------:|------:|---------------|:----------:|---------:|
|    10 |     9 | Basic         |     6      |    6.21  |
|    10 |     9 | SecondarySort |     6      |    5.64  |
|    50 |    49 | Basic         |     8      |    7.79  |
|    50 |    49 | SecondarySort |     8      |    7.74  |
|   100 |    99 | Basic         |     9      |    9.40  |
|   100 |    99 | SecondarySort |     9      |    9.17  |
|   200 |   199 | Basic         |    10      |   10.09  |
|   200 |   199 | SecondarySort |    10      |   11.01  |
|   500 |   499 | Basic         |    12      |   13.19  |
|   500 |   499 | SecondarySort |    12      |   17.75  |

**Key findings:**

- **Iterations grow as approximately O(log₂ n)** for chain graphs: 6 iterations for n=10, 8 for n=50, 9 for n=100, 10 for n=200, 12 for n=500. This matches the theoretical bound — CCF's propagation mechanism approximately doubles the reach of minimum IDs each iteration, yielding ⌈log₂(d)⌉ + c iterations for a graph of diameter d.
- **This is the worst-case topology for CCF**, since chain graphs have diameter d = n−1 (maximum possible for n nodes). The paper acknowledges this as a limitation compared to CC-MR, which achieves O(3 log d) iterations with a lower constant [1, Section II].
- **Runtime is directly proportional to iteration count** in this regime, since per-iteration data volume is minimal (sparse graph, few edges).
- At n=500, the **Basic variant outperforms SecondarySort** (13.2s vs 17.8s). The sorting overhead in the secondary sort variant is not amortised when adjacency lists are small (each node has at most 2 neighbours in a chain), confirming the paper's guidance that secondary sort is advantageous only for large components (50K+ nodes) [1, Section III].

### 3.4 Experiment 3: Cluster Graphs (Multiple Components)

| Clusters | Nodes/Cluster | Inter-edges | Algorithm     | Iterations | Time (s) | Components |
|---------:|--------------:|:-----------:|---------------|:----------:|---------:|-----------:|
|    5     |      20       |      0      | Basic         |     6      |    7.69  |     5      |
|    5     |      20       |      0      | SecondarySort |     6      |    6.07  |     5      |
|    5     |      20       |      4      | Basic         |     7      |    6.98  |     2      |
|    5     |      20       |      4      | SecondarySort |     7      |    6.75  |     2      |
|   10     |      50       |      0      | Basic         |     7      |    7.34  |    10      |
|   10     |      50       |      0      | SecondarySort |     7      |    7.34  |    10      |
|   10     |      50       |      9      | Basic         |     9      |    8.55  |     4      |
|   10     |      50       |      9      | SecondarySort |     9      |   10.19  |     4      |
|   20     |      50       |      0      | Basic         |     7      |    6.99  |    20      |
|   20     |      50       |      0      | SecondarySort |     7      |    7.76  |    20      |
|   20     |      50       |     19      | Basic         |    11      |   10.60  |     4      |
|   20     |      50       |     19      | SecondarySort |    11      |   10.12  |     4      |

**Key findings:**

- **Isolated clusters converge quickly** (6–7 iterations), since each cluster has a small internal diameter and convergence proceeds independently within each component.
- **Adding inter-cluster edges increases iterations** (from 7 to 11 for 20 clusters), because merging clusters creates larger effective components with greater diameter.
- **Component detection is correct across all configurations**: with 0 inter-edges, detected components equals cluster count; adding bridges merges clusters as expected.
- Both algorithm variants perform comparably at this scale, with differences within the noise margin of local Spark execution.

### 3.5 Comparison with Paper Results

The original paper [1] reports results on the web-google dataset (875K nodes, 5.1M edges):

| Algorithm | Iterations | Runtime (s) |
|-----------|:----------:|------------:|
| PEGASUS   |     16     |       2,403 |
| CC-MR     |      8     |         224 |
| CCF       |     11     |         256 |

Our experiments on random graphs of similar density show 5–6 iterations, consistent with the paper's observation that real-world graphs have small diameters. The higher iteration count (11) on web-google reflects its larger diameter compared to synthetic random graphs. The paper's runtime advantage of CCF over PEGASUS (10× faster) aligns with CCF's simpler per-iteration computation.

---

## 4. Comments on Experimental Analysis

### 4.1 Strengths of the CCF Algorithm

1. **Algorithmic simplicity.** CCF requires only two MapReduce jobs (CCF-Iterate + CCF-Dedup) iterated to convergence. This simplicity translates to ease of implementation, debugging, and maintenance — a significant practical advantage for production systems.

2. **Favourable iteration count on real-world graphs.** As Kang et al. [3] demonstrate, real-world networks (social, web, biological) exhibit small diameters. Our random graph experiments confirm that CCF converges in 5–6 iterations regardless of graph size (50–5,000 nodes), and the paper demonstrates 11 iterations on the 875K-node web-google dataset and 13 iterations on a 6B-node production graph. This makes CCF well-suited for the graphs most commonly encountered in practice.

3. **Proven scalability.** The original paper demonstrates CCF on a graph with ~6 billion nodes and ~92 billion edges (7 hours on 80 nodes). To our knowledge, this remains one of the largest publicly reported connected component computations, validating CCF's suitability for industrial-scale graph processing.

4. **Memory-efficient variant.** The secondary sort version (Figure 3) enables processing of graphs with very large connected components (millions of nodes) by avoiding the need to store all reducer values in memory — critical for production deployments where the largest component can contain 50M+ nodes [1].

5. **Deduplication improves efficiency.** The CCF-Dedup step, while adding one extra MapReduce job per iteration, reduces the input volume for subsequent iterations by eliminating redundant pairs. This is a practical optimisation that reduces I/O and shuffle costs.

6. **Natural fit for MapReduce/Spark.** The algorithm uses only standard key-value operations (map, groupByKey, emit) without complex data structures, graph partitioning schemes, or inter-node messaging protocols, making it straightforward to implement on any MapReduce-compatible framework.

### 4.2 Weaknesses and Limitations

1. **Iteration complexity is O(log d), bounded by graph diameter.** For high-diameter graphs (chains, trees, sparse lattices), CCF requires substantially more iterations than for small-world graphs. Our chain graph experiments show 12 iterations for 500 nodes (diameter 499), compared to 6 iterations for random graphs of equivalent size. Algorithms such as CC-MR [4] achieve tighter bounds of O(3 log d) iterations, potentially outperforming CCF on high-diameter graphs.

2. **Per-iteration overhead in distributed settings.** Each iteration involves a full MapReduce shuffle cycle: serialisation, network transfer, disk spill, deserialisation. On distributed clusters, each iteration incurs job scheduling and initialisation overhead (the paper notes this as the primary reason CC-MR's 8-iteration result outperforms CCF's 11-iteration result in wall-clock time on web-google). Reducing iteration count therefore has a multiplicative effect on runtime.

3. **Secondary sort approximation in Spark.** Our Spark implementation uses `groupByKey().mapValues(sorted(...))` to simulate secondary sort. This collects all values into memory before sorting, negating the O(1) memory advantage of the Hadoop secondary sort mechanism. A production Spark implementation should use `repartitionAndSortWithinPartitions` with a custom partitioner to achieve true streaming sort behaviour.

4. **Reducer skew on power-law graphs.** Real-world graphs often follow power-law degree distributions, where a small number of hub nodes have very high degree. In CCF-Iterate, these hubs produce large adjacency lists concentrated on a single reducer, creating load imbalance. The paper does not address this concern, and our synthetic experiments (with relatively uniform degree distributions) do not exercise this weakness.

5. **Duplicate pair generation.** CCF-Iterate can emit the same *(value, min)* pair from multiple reducers within a single iteration, necessitating the CCF-Dedup step. While dedup is efficient (a single MapReduce job), it represents additional I/O and computation that more sophisticated algorithms might avoid through smarter pair management.

6. **No directed graph support.** The algorithm assumes undirected graphs (the mapper emits both directions of each edge). Finding *strongly connected components* in directed graphs requires fundamentally different algorithms (e.g., forward-backward reachability).

7. **Local-mode experimental limitations.** Our experiments run on a single machine, where Spark's per-iteration overhead (~5s) dominates total runtime. This makes it difficult to observe the true scaling characteristics of the algorithm. Meaningful runtime comparisons between Basic and SecondarySort would require graphs with 100K+ nodes on a multi-node cluster.

### 4.3 RDD Implementation Considerations

The translation from Hadoop MapReduce to Spark RDDs introduces several considerations:

- **`groupByKey` vs `reduceByKey`**: We use `groupByKey` in CCF-Iterate because the reduce logic requires access to all values simultaneously (to find the minimum and then emit pairs). Unlike simple aggregations where `reduceByKey` is preferred for its combiner optimisation, CCF's reduce logic is inherently a "group-and-process" pattern.
- **Caching strategy**: Between iterations, the deduplicated RDD is cached in memory (`.cache()`) and the previous iteration's RDD is unpersisted. This prevents recomputation but increases memory pressure — a trade-off that should be tuned based on cluster resources.
- **Accumulator semantics**: Spark accumulators (used for the NewPair counter in Scala) are only guaranteed to be accurate when updated inside *actions*, not *transformations*. We force evaluation with `.count()` before reading the accumulator value.
- **Convergence check**: A `pairs_rdd.count()` action is required each iteration to materialise the RDD and trigger the counter update. This adds a small overhead but is necessary for correctness.

---

## 5. Conclusion

We have implemented and experimentally evaluated the CCF (Connected Component Finder) algorithm [1] in both Python (PySpark) and Scala, faithfully translating the three core algorithms — CCF-Iterate Basic (Figure 2), CCF-Iterate with Secondary Sort (Figure 3), and CCF-Dedup (Figure 4) — from Hadoop MapReduce to the Spark RDD framework.

Our experiments on synthetic graphs confirm the key findings of the original paper:

1. **On random and clustered graphs** (which model real-world network topologies), CCF converges in 5–7 iterations regardless of graph size, validating the paper's claim that real-world graphs have small effective diameters.
2. **On chain graphs** (worst case), iteration count grows logarithmically with diameter, reaching 12 iterations for a 500-node chain — consistent with the theoretical O(log d) bound.
3. **Both algorithm variants produce identical results** in terms of correctness and iteration count, differing only in per-reducer memory behaviour.
4. **The Basic variant is preferable** for moderate-scale graphs due to lower per-iteration overhead, while the SecondarySort variant becomes essential for graphs with very large connected components (millions of nodes).

The CCF algorithm's principal strength lies in its simplicity and proven scalability to billion-node graphs. Its principal weakness is sensitivity to graph diameter, where algorithms with tighter iteration bounds (e.g., CC-MR) may be preferred.

For future work, we recommend: (a) evaluating on larger graphs (100K+ nodes) on a multi-node Spark cluster to observe true distributed scaling behaviour; (b) implementing true secondary sort via `repartitionAndSortWithinPartitions` for production-grade memory efficiency; and (c) comparing against Spark's built-in GraphX `connectedComponents()` implementation, which uses the Pregel abstraction.

---

## References

[1] H. Kardes, S. Agrawal, X. Wang, and A. Sun, "CCF: Fast and Scalable Connected Component Computation in MapReduce," *Data Research, inome Inc.*, Bellevue, WA, USA.

[2] P. Erdős and A. Rényi, "On the evolution of random graphs," *Publications of the Mathematical Institute of the Hungarian Academy of Sciences*, vol. 5, pp. 17–61, 1960.

[3] U. Kang, C. Tsourakakis, and C. Faloutsos, "PEGASUS: mining peta-scale graphs," *Knowledge and Information Systems*, vol. 27, no. 2, pp. 303–325, 2011.

[4] T. Seidl, B. Boden, and S. Fries, "CC-MR: Finding connected components in huge graphs with MapReduce," in *Machine Learning and Knowledge Discovery in Databases*, Springer, 2012, vol. 7523, pp. 458–473.

[5] J. Lin and C. Dyer, *Data-Intensive Text Processing with MapReduce*, Morgan & Claypool Publishers, 2010.

---

## Appendix A: Python Implementation (PySpark)

### A.1 CCF Core Algorithm (`ccf_connected_components.py`)

```python
"""
CCF: Connected Component Finder using MapReduce
Uses PySpark for the MapReduce framework.
"""

from pyspark import SparkContext, SparkConf


# --- Figure 2: CCF-Iterate (Basic Version) ---

def ccf_iterate(pairs_rdd):
    # Map phase: emit both directions to build adjacency lists
    mapped = pairs_rdd.flatMap(lambda pair: [
        (pair[0], pair[1]),
        (pair[1], pair[0])
    ])

    # Reduce phase: group by key, find min, emit new pairs
    grouped = mapped.groupByKey()

    def reduce_fn(key_values):
        key, values_iter = key_values
        value_list = list(values_iter)
        min_val = key
        for value in value_list:
            if value < min_val:
                min_val = value

        results = []
        new_pairs = 0

        if min_val < key:
            results.append((key, min_val))
            for value in value_list:
                if value != min_val:
                    new_pairs += 1
                    results.append((value, min_val))

        return results, new_pairs

    reduced = grouped.map(reduce_fn).cache()
    output_rdd = reduced.flatMap(lambda x: x[0])
    new_pair_count = reduced.map(lambda x: x[1]).reduce(lambda a, b: a + b)
    reduced.unpersist()
    return output_rdd, new_pair_count


# --- Figure 3: CCF-Iterate with Secondary Sorting ---

def ccf_iterate_secondary_sort(pairs_rdd):
    mapped = pairs_rdd.flatMap(lambda pair: [
        (pair[0], pair[1]),
        (pair[1], pair[0])
    ])

    # Sort values to simulate secondary sort (note: collects into memory)
    grouped = mapped.groupByKey().mapValues(lambda vals: sorted(vals))

    def reduce_fn(key_values):
        key, sorted_values = key_values
        results = []
        new_pairs = 0

        if not sorted_values:
            return results, new_pairs

        min_value = sorted_values[0]

        if min_value < key:
            results.append((key, min_value))
            for value in sorted_values[1:]:
                new_pairs += 1
                results.append((value, min_value))

        return results, new_pairs

    reduced = grouped.map(reduce_fn).cache()
    output_rdd = reduced.flatMap(lambda x: x[0])
    new_pair_count = reduced.map(lambda x: x[1]).reduce(lambda a, b: a + b)
    reduced.unpersist()
    return output_rdd, new_pair_count


# --- Figure 4: CCF-Dedup ---

def ccf_dedup(pairs_rdd):
    return pairs_rdd.distinct()


# --- Full CCF Pipeline ---

def find_connected_components(sc, edges, use_secondary_sort=False,
                              max_iterations=100):
    iterate_fn = (ccf_iterate_secondary_sort if use_secondary_sort
                  else ccf_iterate)

    pairs_rdd = sc.parallelize(edges)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        output_rdd, new_pair_count = iterate_fn(pairs_rdd)
        pairs_rdd = ccf_dedup(output_rdd)
        pairs_rdd.cache()
        pairs_rdd.count()  # force evaluation

        if new_pair_count == 0:
            break

    return pairs_rdd


# --- Example Driver ---

if __name__ == "__main__":
    conf = SparkConf().setAppName("CCF").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    edges = [
        ("A", "B"), ("B", "D"), ("D", "E"),
        ("A", "C"), ("A", "E"), ("F", "G"), ("F", "H"),
    ]

    components = find_connected_components(sc, edges)
    for node, comp_id in sorted(components.collect()):
        print(f"  {node} -> {comp_id}")

    sc.stop()
```

### A.2 Experiment Script (`ccf_experiments.py`)

```python
"""
CCF Experiments: Benchmarking on synthetic graphs of increasing size.
"""

import time, random, csv
from pyspark import SparkContext, SparkConf


def generate_chain_graph(n):
    """Chain: 0-1-2-...-n-1. Diameter = n-1."""
    return [(str(i), str(i + 1)) for i in range(n - 1)]


def generate_random_graph(n_nodes, n_edges, seed=42):
    """Erdos-Renyi style random graph."""
    rng = random.Random(seed)
    edges = set()
    while len(edges) < n_edges:
        a, b = rng.randint(0, n_nodes-1), rng.randint(0, n_nodes-1)
        if a != b:
            edges.add((str(min(a, b)), str(max(a, b))))
    return list(edges)


def generate_cluster_graph(n_clusters, nodes_per_cluster, inter_edges=0,
                           seed=42):
    """Multiple dense clusters with optional inter-cluster bridges."""
    rng = random.Random(seed)
    edges = []
    for c in range(n_clusters):
        base = c * nodes_per_cluster
        for i in range(nodes_per_cluster - 1):
            edges.append((str(base + i), str(base + i + 1)))
            if i + 2 < nodes_per_cluster:
                edges.append((str(base + i), str(base + i + 2)))
    for _ in range(inter_edges):
        c1, c2 = rng.sample(range(n_clusters), 2)
        n1 = c1 * nodes_per_cluster + rng.randint(0, nodes_per_cluster - 1)
        n2 = c2 * nodes_per_cluster + rng.randint(0, nodes_per_cluster - 1)
        edges.append((str(n1), str(n2)))
    return edges


def run_ccf(sc, edges, iterate_fn, max_iterations=100):
    start = time.time()
    pairs_rdd = sc.parallelize(edges)
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        output_rdd, new_pair_count = iterate_fn(pairs_rdd)
        pairs_rdd = output_rdd.distinct()  # CCF-Dedup
        pairs_rdd.cache()
        pairs_rdd.count()
        if new_pair_count == 0:
            break
    elapsed = time.time() - start
    num_components = pairs_rdd.map(lambda x: x[1]).distinct().count()
    return iteration, elapsed, num_components


# [Graph sizes tested: random 50-5000 nodes, chain 10-500, clusters 5-20]
# Results saved to experiment_results.csv
```

---

## Appendix B: Scala Implementation (`CCFConnectedComponents.scala`)

```scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator

object CCFConnectedComponents {

  // --- Figure 2: CCF-Iterate (Basic Version) ---
  def ccfIterate(
      pairsRDD: RDD[(String, String)],
      newPairCounter: LongAccumulator
  ): RDD[(String, String)] = {
    val mapped = pairsRDD.flatMap { case (key, value) =>
      Seq((key, value), (value, key))
    }
    mapped.groupByKey().flatMap { case (key, valuesIter) =>
      val valueList = valuesIter.toList
      var min = key
      for (value <- valueList) if (value < min) min = value
      if (min < key) {
        val results = scala.collection.mutable.ListBuffer((key, min))
        for (value <- valueList) if (value != min) {
          newPairCounter.add(1)
          results += ((value, min))
        }
        results.toList
      } else List.empty
    }
  }

  // --- Figure 3: CCF-Iterate with Secondary Sorting ---
  def ccfIterateSecondarySort(
      pairsRDD: RDD[(String, String)],
      newPairCounter: LongAccumulator
  ): RDD[(String, String)] = {
    val mapped = pairsRDD.flatMap { case (key, value) =>
      Seq((key, value), (value, key))
    }
    mapped.groupByKey().flatMap { case (key, valuesIter) =>
      val sortedValues = valuesIter.toList.sorted
      if (sortedValues.nonEmpty && sortedValues.head < key) {
        val minValue = sortedValues.head
        val results = scala.collection.mutable.ListBuffer((key, minValue))
        for (value <- sortedValues.tail) {
          newPairCounter.add(1)
          results += ((value, minValue))
        }
        results.toList
      } else List.empty
    }
  }

  // --- Figure 4: CCF-Dedup ---
  def ccfDedup(pairsRDD: RDD[(String, String)]): RDD[(String, String)] = {
    pairsRDD.map(pair => (pair, null))
      .reduceByKey((a, _) => a)
      .map(_._1)
  }

  // --- Full Pipeline ---
  def findConnectedComponents(
      sc: SparkContext, edges: Seq[(String, String)],
      useSecondarySort: Boolean = false, maxIterations: Int = 100
  ): RDD[(String, String)] = {
    var pairsRDD = sc.parallelize(edges)
    var converged = false
    var iteration = 0
    while (iteration < maxIterations && !converged) {
      iteration += 1
      val counter = sc.longAccumulator("NewPairCounter")
      val output = if (useSecondarySort)
        ccfIterateSecondarySort(pairsRDD, counter)
      else ccfIterate(pairsRDD, counter)
      val deduped = ccfDedup(output)
      deduped.cache()
      deduped.count()
      pairsRDD.unpersist()
      pairsRDD = deduped
      if (counter.value == 0) converged = true
    }
    pairsRDD
  }
}
```
