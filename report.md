# Implementation and Analysis of CCF Algorithm in MapReduce

### Big Data Project
### David W. Fong, Dileep Kumar Malhi

---

## 1. Description of Solution

### 1.1 Problem

Finding connected components in a graph is a fundamental problem in graph theory with applications in social network analysis, record linkage, image segmentation, and data mining [1]. Given an undirected graph *G = (V, E)*, the objective is to partition *V* into disjoint sets *C = {C₁, C₂, ..., Cₙ}* such that for each component *Cᵢ*, there exists a path between any two vertices *vₖ, vₗ ∈ Cᵢ*, and no path exists between vertices in distinct components.

As graph sizes have grown to billions of nodes and edges in domains such as social networks, web graphs, and entity resolution systems, sequential algorithms are no longer feasible. Distributed computing frameworks such as MapReduce and Apache Spark allow graph algorithms to scale across clusters of commodity machines.

### 1.2 Method

We adopt the Connected Component Finder (CCF) algorithm proposed by Kardes et al. [1], which identifies connected components through iterative MapReduce jobs. The algorithm represents each component by the smallest node ID it contains. Through repeated iterations of two MapReduce jobs, CCF-Iterate and CCF-Dedup, minimum node IDs are propagated along edges until every node in a component maps to the same minimum ID.

The algorithm takes as input an edge list and produces a mapping from each node to its component identifier (the smallest node ID in its component). The iterative process terminates when no new pairs are generated in a given iteration, as tracked by a global counter.

### 1.3 Implementation

We implement the CCF algorithm using Apache Spark (PySpark 4.0.1 and Scala) with Resilient Distributed Datasets (RDDs) as the distributed data abstraction. RDDs provide immutability and fault tolerance through lineage-based recovery, lazy evaluation where transformations are deferred until an action (e.g., `count()`, `collect()`) triggers execution, and in-memory caching of intermediate results across iterations via `.cache()`.

The MapReduce paradigm translates directly to the RDD API: the map phase corresponds to `flatMap` transformations, and the reduce phase corresponds to `groupByKey` followed by per-key aggregation. The iterative structure of CCF is implemented as a `while` loop that applies CCF-Iterate and CCF-Dedup until convergence.

### 1.4 Comparison with Original Paper

The original CCF paper [1] was implemented in Java using Hadoop MapReduce and demonstrated on a graph with ~6 billion nodes and ~92 billion edges on an 80-node Hadoop cluster. Our implementation preserves the algorithmic logic while adapting to the Spark RDD API. The differences from the Hadoop version are: (1) RDD-based data flow replaces HDFS file-based intermediate storage between iterations; (2) intermediate RDDs are cached in memory between iterations, reducing I/O overhead; (3) the NewPair counter is implemented via manual counting in the reduce function (Python) or a Spark `LongAccumulator` (Scala), rather than Hadoop's global counter mechanism; (4) CCF-Dedup uses Spark's `distinct()` (Python) or `reduceByKey` on composite keys (Scala), which is semantically equivalent to the paper's composite-key deduplication pattern.

The secondary sort variant (Section 2.2) in the paper uses Hadoop's secondary sort mechanism with custom partitioning to deliver values to the reducer in sorted order without loading them into memory. In our Spark implementation, we approximate this with `groupByKey().mapValues(sorted(...))`, which collects values into memory before sorting. Our secondary sort variant therefore does not achieve the O(1) memory guarantee of the Hadoop implementation at the reducer level. A true Spark equivalent would require `repartitionAndSortWithinPartitions` with a custom partitioner.

---

## 2. Algorithms

### 2.1 Overview

The CCF module consists of two MapReduce jobs executed iteratively: CCF-Iterate, which propagates minimum node IDs along edges, and CCF-Dedup, which removes duplicate pairs between iterations. The process repeats until no new pairs are generated, tracked by a global counter `NewPair`. The output is a set of *(node, componentID)* pairs where `componentID` is the smallest node ID in the component.

### 2.2 CCF-Iterate, Basic Version 

The basic CCF-Iterate algorithm, whose implementation is shown in Figure 2 of [1], constructs adjacency lists and propagates minimum IDs.

**Map Phase:**
For each input pair *(key, value)*, emit both *(key, value)* and *(value, key)*. This ensures each node appears as a key with all its neighbours as values, effectively constructing a bidirectional adjacency list.

**Reduce Phase:**
For each key (node) with its grouped values (neighbours):
1. Find the minimum value `min` across all values and the key itself.
2. If `min < key` (a smaller ID exists in the neighbourhood):
   - Emit *(key, min)*, mapping the key to the smaller ID.
   - For each value ≠ min, emit *(value, min)*, propagating the minimum to all neighbours.
   - Increment the `NewPair` counter for each such emission.
3. If `min ≥ key`, emit nothing (this node is already the local minimum).

**Space complexity:** O(N) per reducer, where N is the size of the adjacency list, since all values must be stored in a list to iterate twice (once to find the minimum, once to emit pairs).

**RDD Translation:**
```python
mapped = pairs_rdd.flatMap(lambda pair: [(pair[0], pair[1]), (pair[1], pair[0])])
grouped = mapped.groupByKey()
# reduce_fn iterates values twice: find min, then emit pairs
```

### 2.3 CCF-Iterate, Secondary Sort Version

The secondary sort variant, whose implementation is shown in Figure 3 of [1], improves memory efficiency by delivering values to the reducer in sorted order, so the first value is guaranteed to be the minimum.

**Map Phase:** Identical to the basic version.

**Reduce Phase:**
1. The first value from the sorted iterator is `minValue`.
2. If `minValue < key`:
   - Emit *(key, minValue)*.
   - For each remaining value in the iterator, emit *(value, minValue)* and increment `NewPair`.
3. If `minValue ≥ key`, emit nothing.

**Space complexity:** O(1) per reducer in the original Hadoop implementation, where values are streamed from disk in sorted order. In our Spark implementation, this is approximated with an explicit sort after `groupByKey`, which still requires O(N) memory (see Section 1.4).

**RDD Translation:**
```python
grouped = mapped.groupByKey().mapValues(lambda vals: sorted(vals))
# reduce_fn: first element of sorted list is minValue; single pass
```

### 2.4 CCF-Dedup

CCF-Iterate may emit the same pair multiple times within a single iteration. CCF-Dedup, whose implementation is shown in Figure 4 of [1], removes these duplicates to reduce the input size for the next iteration.

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

The algorithm converges when no new pairs are generated in an iteration. At each iteration, minimum IDs propagate at least one hop further, so convergence is guaranteed in at most *d* iterations where *d* is the graph diameter. In practice, the number of iterations is much smaller than *d* because each CCF-Iterate step can propagate minimum IDs across multiple hops: when a node receives a new minimum, that minimum is immediately forwarded to all its neighbours in the same iteration. The paper does not provide a formal proof of the iteration bound, but empirically the iteration count grows logarithmically with the diameter. Our experiments confirm this (see Section 3.3).

### 2.6 Correctness

The algorithm correctly identifies connected components for three reasons. First, component IDs can only decrease (each node maps to the minimum in its neighbourhood), which guarantees convergence. Second, the map phase emits edges in both directions, ensuring bidirectional reachability. Third, the iteration continues until no new pairs are generated, so every node converges to the global minimum ID of its component.

---

## 3. Experimental Analysis

### 3.1 Experimental Setup

All experiments were conducted on a single machine (Apple M-series, 8 cores) using PySpark 4.0.1 in local mode (`local[*]`). While this does not replicate a true distributed cluster, it enables controlled comparison of algorithmic behaviour (iteration counts, convergence patterns) across graph topologies.

Both algorithm variants, Basic (Figure 2) and SecondarySort (Figure 3), were evaluated on three classes of synthetic graphs: random graphs (Erdős-Rényi model) with increasing size from 50 to 5,000 nodes and edge density of approximately 3x the node count; chain graphs as linear paths of 10 to 500 nodes representing worst-case diameter; and cluster graphs consisting of multiple dense clusters with optional inter-cluster edges to control component count.

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

The iteration count is nearly constant at 5 to 6 regardless of graph size (50 to 5,000 nodes), consistent with the paper's observation that real-world graphs have small effective diameters [1, Section IV] and the small-world properties of random graphs [4]. Runtime increases modestly from 5.0s to 6.8s for a 100x increase in graph size, indicating that at this scale the fixed overhead of Spark job initialisation dominates per-iteration data processing. Both variants yield identical iteration counts, confirming that they implement the same convergence logic and differ only in reducer memory management. All generated graphs form a single connected component; at density 3x the node count (average degree 6) the Erdős-Rényi giant-component threshold (average degree > 1) is comfortably exceeded, though the full-connectivity threshold (average degree ≥ ln n) is only met for smaller graphs in this range (ln 50 ≈ 3.9 < 6, but ln 5000 ≈ 8.5 > 6). The single-component result for larger graphs reflects the specific random seed (42) rather than a guaranteed theoretical threshold.

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

Iterations grow logarithmically with chain length: 6 iterations for n=10, 8 for n=50, 9 for n=100, 10 for n=200, and 12 for n=500. For chain graphs, *d* = n-1, so these values are consistent with approximately log2(d) + c iterations. Note that the CCF paper does not formally prove this bound; we observe it empirically. The logarithmic behaviour arises because each iteration can propagate a minimum ID across multiple hops when intermediate nodes relay it to their neighbours within the same reduce step. Chain graphs are the worst-case topology for CCF since they have the maximum possible diameter for a given node count. The paper acknowledges the diameter dependency as a limitation compared to CC-MR, which achieves O(3 log d) iterations [1, Section II]. Runtime is directly proportional to iteration count in this regime because per-iteration data volume is minimal. At n=500, the Basic variant is faster than SecondarySort (13.2s vs 17.8s). The sorting overhead in the secondary sort variant is not amortised when adjacency lists are small (each node has at most 2 neighbours in a chain), confirming the paper's guidance that secondary sort is advantageous only for large components [1, Section III].

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

Isolated clusters converge in 6 to 7 iterations since each cluster has a small internal diameter and convergence proceeds independently within each component. Adding inter-cluster edges increases the iteration count (from 7 to 11 for 20 clusters) because merging clusters creates larger effective components with greater diameter. Component detection is correct across all configurations: with 0 inter-edges, detected components equal the cluster count, and adding bridges merges clusters as expected. Both algorithm variants perform comparably at this scale, with differences within the noise margin of local Spark execution.

### 3.5 Python vs Scala Runtime Comparison

Both implementations were run on the same machine (Apple M-series, 8 cores, local Spark mode) using PySpark 4.0.1 / Spark 4.0.1 (Scala 2.13.16, Java 17). The *Iters* column below shows the Scala run's iteration count. For chain graphs, which are deterministic, Python and Scala always produce identical iteration counts and component assignments. For random and cluster-with-inter-edges graphs, the Python and Scala random number generators differ (Java's `java.util.Random` vs Python's `random.Random`), so at the same seed they generate different edge sets — iteration counts and component counts can therefore differ by small amounts between the two implementations for those cases. Only runtimes are compared.

**Experiment 1: Random Graphs**

| Nodes | Edges | Algorithm | Iters | Python (s) | Scala (s) | Speedup |
|------:|------:|-----------|:-----:|-----------:|----------:|--------:|
|    50 |   100 | Basic         | 5 |  5.90 | 1.64 | 3.6× |
|    50 |   100 | SecondarySort | 5 |  5.06 | 0.46 | 11.0× |
|   100 |   300 | Basic         | 5 |  5.09 | 0.48 | 10.6× |
|   100 |   300 | SecondarySort | 5 |  5.23 | 0.39 | 13.4× |
|   500 | 1,500 | Basic         | 5 |  5.29 | 0.70 | 7.6× |
|   500 | 1,500 | SecondarySort | 6 |  5.67 | 0.60 | 9.5× |
| 1,000 | 3,000 | Basic         | 5 |  6.89 | 0.45 | 15.3× |
| 1,000 | 3,000 | SecondarySort | 6 |  6.26 | 0.42 | 14.9× |
| 2,000 | 6,000 | Basic         | 6 |  6.67 | 0.61 | 10.9× |
| 2,000 | 6,000 | SecondarySort | 6 |  5.76 | 0.61 | 9.4× |
| 5,000 |15,000 | Basic         | 6 |  6.58 | 0.84 | 7.8× |
| 5,000 |15,000 | SecondarySort | 6 |  6.78 | 0.78 | 8.7× |

**Experiment 2: Chain Graphs**

| Nodes | Edges | Algorithm | Iters | Python (s) | Scala (s) | Speedup |
|------:|------:|-----------|:-----:|-----------:|----------:|--------:|
|    10 |     9 | Basic         |  6 |  6.21 | 0.22 | 28.2× |
|    10 |     9 | SecondarySort |  6 |  5.64 | 0.19 | 29.7× |
|    50 |    49 | Basic         |  8 |  7.79 | 0.31 | 25.1× |
|    50 |    49 | SecondarySort |  8 |  7.74 | 0.30 | 25.8× |
|   100 |    99 | Basic         |  9 |  9.40 | 0.40 | 23.5× |
|   100 |    99 | SecondarySort |  9 |  9.17 | 0.39 | 23.5× |
|   200 |   199 | Basic         | 10 | 10.09 | 0.51 | 19.8× |
|   200 |   199 | SecondarySort | 10 | 11.01 | 0.51 | 21.6× |
|   500 |   499 | Basic         | 12 | 13.19 | 1.26 | 10.5× |
|   500 |   499 | SecondarySort | 12 | 17.75 | 1.26 | 14.1× |

**Experiment 3: Cluster Graphs**

| Clusters | Nodes/Cluster | Inter-edges | Algorithm | Iters | Python (s) | Scala (s) | Speedup |
|---------:|--------------:|:-----------:|-----------|:-----:|-----------:|----------:|--------:|
|  5 |  20 |  0 | Basic         |  6 |  7.69 | 0.21 | 36.6× |
|  5 |  20 |  0 | SecondarySort |  6 |  6.07 | 0.21 | 28.9× |
|  5 |  20 |  4 | Basic         |  8 |  6.98 | 0.27 | 25.9× |
|  5 |  20 |  4 | SecondarySort |  8 |  6.75 | 0.26 | 26.0× |
| 10 |  50 |  0 | Basic         |  7 |  7.34 | 0.47 | 15.6× |
| 10 |  50 |  0 | SecondarySort |  7 |  7.34 | 0.45 | 16.3× |
| 10 |  50 |  9 | Basic         |  9 |  8.55 | 0.57 | 15.0× |
| 10 |  50 |  9 | SecondarySort |  9 | 10.19 | 0.53 | 19.2× |
| 20 |  50 |  0 | Basic         |  7 |  6.99 | 0.56 | 12.5× |
| 20 |  50 |  0 | SecondarySort |  7 |  7.76 | 0.60 | 12.9× |
| 20 |  50 | 19 | Basic         | 10 | 10.60 | 0.70 | 15.1× |
| 20 |  50 | 19 | SecondarySort | 11 | 10.12 | 0.88 | 11.5× |

**Analysis.** Scala is consistently 8–37× faster than Python across all experiments. The largest speedups occur on small, sparse graphs (chain n=10: ~29×, 5-cluster isolated: ~37×), where Python's fixed per-job overhead (~5s baseline from PySpark initialisation and Python-JVM serialisation) dominates total runtime. On larger graphs where actual computation increases (random n=5000: ~8×, chain n=500: ~10–14×), the speedup narrows but remains substantial.

The performance gap has two sources. First, Python incurs a serialisation cost at every RDD transformation: Python objects are pickled, sent to the JVM, processed, and results are unpickled back. Scala operates natively on the JVM, eliminating this round-trip entirely. Second, Scala's static typing allows the JIT compiler to optimise the inner loops of the reduce function (adjacency list traversal, min finding) more aggressively than CPython can.

For chain graphs (deterministic), both implementations produce identical iteration counts and component assignments, confirming semantic equivalence of the core algorithm. For random and cluster graphs with inter-edges, small differences in iteration counts and component counts arise from the different RNG implementations between Java and Python; the algorithms are structurally equivalent but operate on slightly different input graphs at the same seed.

### 3.6 Comparison with Original Paper

The original paper [1] reports results on the web-google dataset (875K nodes, 5.1M edges):

| Algorithm | Iterations | Runtime (s) |
|-----------|:----------:|------------:|
| PEGASUS   |     16     |       2,403 |
| CC-MR     |      8     |         224 |
| CCF       |     11     |         256 |

Direct runtime comparison between our results and the paper's is not meaningful: the paper uses Hadoop on a 50-node cluster with the 875K-node web-google dataset, while our experiments use PySpark in local mode on graphs up to 5,000 nodes. However, the iteration counts are comparable. Our random graphs converge in 5 to 6 iterations, while web-google requires 11, reflecting its larger diameter. The relative ordering of algorithms (PEGASUS=16 > CCF=11 > CC-MR=8 in iteration count) is consistent with the following characterisation: PEGASUS requires O(d) iterations, making it slowest on any graph. CCF and CC-MR both exhibit logarithmic convergence in practice, but CC-MR converges faster empirically (8 vs 11 iterations) because it has a formal proven bound of at most 3 log d iterations [1, Section II], whereas CCF has no formal proof of its logarithmic behaviour — the O(log d) convergence is empirical only (as noted in Section 2.5). CCF's formal worst-case bound remains O(d).

---

## 4. Comments on Experimental Analysis

### 4.1 Strengths of the CCF Algorithm

CCF uses only two MapReduce jobs and standard key-value operations (map, groupByKey, emit) without graph partitioning schemes or inter-node messaging protocols. This makes it portable across MapReduce-compatible frameworks and easy to debug, since each iteration's input and output are plain key-value pair files that can be inspected directly.

The low iteration count on small-diameter graphs (as confirmed in our experiments and by Kang et al. [3]) means the total number of MapReduce rounds is small in practice, which is the dominant cost factor in distributed settings. The availability of two reducer variants allows the implementation to be adapted to the expected component size: the basic variant avoids sorting overhead for moderate components, while the secondary sort variant caps reducer memory usage for components with millions of nodes [1]. The CCF-Dedup step adds one MapReduce job per iteration but reduces shuffle volume in subsequent iterations, which is a net benefit when duplicate pairs are numerous.

### 4.2 Weaknesses and Limitations

The number of iterations depends on graph diameter. Empirically, iterations grow as approximately log2(d), but the paper does not provide a formal proof of this bound — CCF's formal worst-case bound is O(d). For high-diameter graphs (chains, trees, sparse lattices), CCF requires more iterations than for small-world graphs. Our chain graph experiments show 12 iterations for 500 nodes (diameter 499), compared to 6 iterations for random graphs of equivalent size. CC-MR [2] provides a formal proven convergence guarantee of at most 3 log d iterations, giving it a tighter worst-case bound than CCF's formal O(d). In practice, CC-MR also converges faster on the same graphs (8 vs 11 iterations on web-google). Each iteration involves a full MapReduce shuffle cycle (serialisation, network transfer, disk spill, deserialisation), so each additional iteration incurs job scheduling and initialisation overhead. The combination of fewer iterations and equivalent per-iteration cost is the primary reason CC-MR's 8-iteration result outperforms CCF's 11-iteration result in wall-clock time on web-google.

Our Spark implementation uses `groupByKey().mapValues(sorted(...))` to simulate secondary sort. This collects all values into memory before sorting, negating the O(1) memory advantage of the Hadoop secondary sort mechanism. A production Spark implementation should use `repartitionAndSortWithinPartitions` with a custom partitioner to achieve true streaming sort behaviour.

Real-world graphs often follow power-law degree distributions where a small number of hub nodes have very high degree. In CCF-Iterate, these hubs produce large adjacency lists concentrated on a single reducer, creating load imbalance. The paper does not address this concern, and our synthetic experiments (with relatively uniform degree distributions) do not exercise this weakness.

CCF-Iterate can also emit the same *(value, min)* pair from multiple reducers within a single iteration, requiring the CCF-Dedup step. While dedup is efficient (a single MapReduce job), it is additional I/O and computation that more sophisticated algorithms avoid. The algorithm also assumes undirected graphs (the mapper emits both directions of each edge); finding strongly connected components in directed graphs requires different algorithms such as forward-backward reachability.

Finally, our experiments run on a single machine where Spark's per-iteration overhead (~5s) dominates total runtime. This makes it difficult to observe the true scaling characteristics of the algorithm. Meaningful runtime comparisons between Basic and SecondarySort would require graphs with 100K+ nodes on a multi-node cluster.

### 4.3 RDD Implementation Considerations

The translation from Hadoop MapReduce to Spark RDDs introduces several considerations. We use `groupByKey` in CCF-Iterate because the reduce logic requires access to all values simultaneously to find the minimum and then emit pairs. Unlike simple aggregations where `reduceByKey` is preferred for its combiner optimisation, CCF's reduce logic is a group-and-process pattern. Between iterations, the deduplicated RDD is cached in memory (`.cache()`) and the previous iteration's RDD is unpersisted. This prevents recomputation but increases memory pressure, a trade-off that should be tuned based on cluster resources. Spark accumulators (used for the NewPair counter in Scala) are only guaranteed to be accurate when updated inside actions, not transformations, so we force evaluation with `.count()` before reading the accumulator value. This `count()` action is required each iteration to materialise the RDD and trigger the counter update.

---

## 5. Conclusion

We have implemented and experimentally evaluated the CCF algorithm [1] in both Python (PySpark) and Scala, translating the three core algorithms (CCF-Iterate Basic, CCF-Iterate with Secondary Sort, and CCF-Dedup) from Hadoop MapReduce to the Spark RDD framework.

Our experiments on synthetic graphs confirm the findings of the original paper. On random and clustered graphs, which model real-world network topologies, CCF converges in 5 to 7 iterations regardless of graph size. On chain graphs (worst case), iteration count grows logarithmically with diameter, reaching 12 iterations for a 500-node chain. Both algorithm variants produce identical results in terms of correctness and iteration count, differing only in per-reducer memory behaviour. The Basic variant is preferable for moderate-scale graphs due to lower per-iteration overhead, while the SecondarySort variant becomes necessary for graphs with very large connected components.

A direct Python vs Scala runtime comparison across all 36 experiment configurations shows Scala to be 8–37× faster. The speedup is largest on small sparse graphs where Python's fixed serialisation overhead (~5s per run) dominates, and narrows on larger graphs where computation time grows. For deterministic (chain) graphs, both implementations produce identical iteration counts and component assignments, confirming core algorithmic equivalence.

CCF is simple to implement and has proven scalability to billion-node graphs. Its main limitation is sensitivity to graph diameter, where algorithms with tighter iteration bounds such as CC-MR may be preferred.

---

## References

[1] H. Kardes, S. Agrawal, X. Wang, and A. Sun, "CCF: Fast and Scalable Connected Component Computation in MapReduce," in *Proc. International Conference on Information and Knowledge Management (CIKM)*, 2014.

[2] T. Seidl, B. Boden, and S. Fries, "CC-MR: Finding connected components in huge graphs with MapReduce," in *Machine Learning and Knowledge Discovery in Databases*, Springer, 2012, vol. 7523, pp. 458–473.

[3] U. Kang, C. Tsourakakis, and C. Faloutsos, "PEGASUS: mining peta-scale graphs," *Knowledge and Information Systems*, vol. 27, no. 2, pp. 303–325, 2011.

[4] P. Erdős and A. Rényi, "On the evolution of random graphs," *Publications of the Mathematical Institute of the Hungarian Academy of Sciences*, vol. 5, pp. 17–61, 1960.

---

## Appendix A: Python Implementation

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

### A.2 Experiment Running Script (`ccf_experiments.py`)

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
    """Erdős-Rényi style random graph."""
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

## Appendix B: Scala Implementation

## B.1 CCF Core Algorithm  (`CCFConnectedComponents.scala`)

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

---

## B.2 Experiment Running Script (`CCFExperiments.scala`)

```scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator

import scala.collection.mutable
import scala.util.Random

object CCFExperiments {

  // ===========================================================================
  // Graph Generators (mirrors ccf_experiments.py)
  // ===========================================================================

  def generateChainGraph(n: Int): Seq[(String, String)] =
    (0 until n - 1).map(i => (i.toString, (i + 1).toString))

  def generateRandomGraph(nNodes: Int, nEdges: Int, seed: Int = 42): Seq[(String, String)] = {
    val rng = new Random(seed)
    val edges = mutable.Set[(String, String)]()
    while (edges.size < nEdges) {
      val a = rng.nextInt(nNodes)
      val b = rng.nextInt(nNodes)
      if (a != b) {
        val lo = math.min(a, b).toString
        val hi = math.max(a, b).toString
        edges.add((lo, hi))
      }
    }
    edges.toSeq
  }

  def generateClusterGraph(nClusters: Int, nodesPerCluster: Int,
                           interEdges: Int = 0, seed: Int = 42): Seq[(String, String)] = {
    val rng = new Random(seed)
    val edges = mutable.ListBuffer[(String, String)]()
    for (c <- 0 until nClusters) {
      val base = c * nodesPerCluster
      for (i <- 0 until nodesPerCluster - 1) {
        edges += ((( base + i).toString, (base + i + 1).toString))
        if (i + 2 < nodesPerCluster)
          edges += (((base + i).toString, (base + i + 2).toString))
      }
    }
    for (_ <- 0 until interEdges) {
      val cs = rng.shuffle((0 until nClusters).toList).take(2)
      val n1 = cs(0) * nodesPerCluster + rng.nextInt(nodesPerCluster)
      val n2 = cs(1) * nodesPerCluster + rng.nextInt(nodesPerCluster)
      edges += ((n1.toString, n2.toString))
    }
    edges.toSeq
  }

  // ===========================================================================
  // CCF Algorithm
  // ===========================================================================

  def ccfIterate(pairsRDD: RDD[(String, String)],
                 counter: LongAccumulator): RDD[(String, String)] = {
    val mapped = pairsRDD.flatMap { case (k, v) => Seq((k, v), (v, k)) }
    mapped.groupByKey().flatMap { case (key, valuesIter) =>
      val valueList = valuesIter.toList
      var min = key
      for (v <- valueList) if (v < min) min = v
      if (min < key) {
        val results = mutable.ListBuffer((key, min))
        for (v <- valueList) if (v != min) { counter.add(1); results += ((v, min)) }
        results.toList
      } else List.empty
    }
  }

  def ccfIterateSecondarySort(pairsRDD: RDD[(String, String)],
                              counter: LongAccumulator): RDD[(String, String)] = {
    val mapped = pairsRDD.flatMap { case (k, v) => Seq((k, v), (v, k)) }
    mapped.groupByKey().flatMap { case (key, valuesIter) =>
      val sorted = valuesIter.toList.sorted
      if (sorted.nonEmpty && sorted.head < key) {
        val minValue = sorted.head
        val results = mutable.ListBuffer((key, minValue))
        for (v <- sorted.tail) { counter.add(1); results += ((v, minValue)) }
        results.toList
      } else List.empty
    }
  }

  def ccfDedup(pairsRDD: RDD[(String, String)]): RDD[(String, String)] =
    pairsRDD.map(p => (p, null)).reduceByKey((a, _) => a).map(_._1)

  // ===========================================================================
  // Experiment Runner
  // ===========================================================================

  case class Result(experiment: String, nodes: Int, edges: Int, algorithm: String,
                    iterations: Int, runtimeSec: Double, components: Int,
                    clusters: Int = 0, interEdges: Int = 0)

  def runCCF(sc: SparkContext, edges: Seq[(String, String)],
             useSecondarySort: Boolean, maxIterations: Int = 100): (Int, Double, Int) = {
    val t0 = System.currentTimeMillis()
    var pairsRDD = sc.parallelize(edges)
    var iteration = 0
    var converged = false
    while (iteration < maxIterations && !converged) {
      iteration += 1
      val counter = sc.longAccumulator("NewPair")
      val output = if (useSecondarySort) ccfIterateSecondarySort(pairsRDD, counter)
                   else ccfIterate(pairsRDD, counter)
      val deduped = ccfDedup(output)
      deduped.cache()
      deduped.count()
      pairsRDD.unpersist()
      pairsRDD = deduped
      if (counter.value == 0) converged = true
    }
    val elapsed = (System.currentTimeMillis() - t0) / 1000.0
    val nComponents = pairsRDD.map(_._2).distinct().count().toInt
    (iteration, elapsed, nComponents)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("CCF-Experiments").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val results = mutable.ListBuffer[Result]()

    // ---- Experiment 1: Random graphs ----
    println("=" * 70)
    println("EXPERIMENT 1: Random graphs of increasing size")
    println("=" * 70)
    println(f"${"Nodes"}%8s ${"Edges"}%8s ${"Algorithm"}%15s ${"Iters"}%6s ${"Time(s)"}%10s ${"Components"}%12s")
    println("-" * 70)

    val randomConfigs = Seq((50, 100), (100, 300), (500, 1500), (1000, 3000), (2000, 6000), (5000, 15000))
    for ((nNodes, nEdges) <- randomConfigs) {
      val edges = generateRandomGraph(nNodes, nEdges)
      for ((name, useSecSort) <- Seq(("Basic", false), ("SecondarySort", true))) {
        val (iters, elapsed, nComp) = runCCF(sc, edges, useSecSort)
        println(f"$nNodes%8d $nEdges%8d $name%15s $iters%6d $elapsed%10.3f $nComp%12d")
        results += Result("random_graph", nNodes, nEdges, name, iters, elapsed, nComp)
      }
    }

    // ---- Experiment 2: Chain graphs ----
    println()
    println("=" * 70)
    println("EXPERIMENT 2: Chain graphs (worst-case diameter)")
    println("=" * 70)
    println(f"${"Nodes"}%8s ${"Edges"}%8s ${"Algorithm"}%15s ${"Iters"}%6s ${"Time(s)"}%10s ${"Components"}%12s")
    println("-" * 70)

    val chainSizes = Seq(10, 50, 100, 200, 500)
    for (n <- chainSizes) {
      val edges = generateChainGraph(n)
      for ((name, useSecSort) <- Seq(("Basic", false), ("SecondarySort", true))) {
        val (iters, elapsed, nComp) = runCCF(sc, edges, useSecSort)
        println(f"$n%8d ${n - 1}%8d $name%15s $iters%6d $elapsed%10.3f $nComp%12d")
        results += Result("chain_graph", n, n - 1, name, iters, elapsed, nComp)
      }
    }

    // ---- Experiment 3: Cluster graphs ----
    println()
    println("=" * 70)
    println("EXPERIMENT 3: Cluster graphs (multiple components)")
    println("=" * 70)
    println(f"${"Clusters"}%9s ${"Nodes/C"}%8s ${"Inter"}%6s ${"Algorithm"}%15s ${"Iters"}%6s ${"Time(s)"}%10s ${"Components"}%12s")
    println("-" * 70)

    val clusterConfigs = Seq((5, 20, 0), (5, 20, 4), (10, 50, 0), (10, 50, 9), (20, 50, 0), (20, 50, 19))
    for ((nClusters, npc, inter) <- clusterConfigs) {
      val edges = generateClusterGraph(nClusters, npc, inter)
      for ((name, useSecSort) <- Seq(("Basic", false), ("SecondarySort", true))) {
        val (iters, elapsed, nComp) = runCCF(sc, edges, useSecSort)
        println(f"$nClusters%9d $npc%8d $inter%6d $name%15s $iters%6d $elapsed%10.3f $nComp%12d")
        results += Result("cluster_graph", nClusters * npc, edges.size, name, iters, elapsed, nComp, nClusters, inter)
      }
    }

    // ---- Write CSV ----
    val csvPath = "experiment_results_scala.csv"
    val pw = new java.io.PrintWriter(csvPath)
    pw.println("experiment,nodes,edges,algorithm,iterations,runtime_sec,components,clusters,inter_edges")
    for (r <- results) {
      pw.println(s"${r.experiment},${r.nodes},${r.edges},${r.algorithm},${r.iterations},${r.runtimeSec},${r.components},${r.clusters},${r.interEdges}")
    }
    pw.close()
    println(s"\nResults saved to $csvPath")

    sc.stop()
  }
}
```
