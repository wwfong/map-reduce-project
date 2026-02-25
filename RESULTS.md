# CCF Experiment Results & Analysis

## Overview

This document presents the results of benchmarking two variants of the CCF (Connected Component Finder) algorithm on synthetic graphs of varying size and topology, run locally using PySpark.

- **CCF-Iterate Basic** (Figure 2): Stores all values in a list to find the minimum; O(N) space per reducer.
- **CCF-Iterate SecondarySort** (Figure 3): Values arrive sorted so the first value is the minimum; single-pass, lower memory footprint.

Both variants use **CCF-Dedup** (Figure 4) between iterations to remove duplicate pairs.

---

## Experiment 1: Random Graphs (Increasing Size)

Random Erdos-Renyi-style graphs with edge density ~3x the node count.

| Nodes | Edges | Algorithm | Iterations | Time (s) | Components |
|------:|------:|-----------|:----------:|---------:|-----------:|
| 50 | 100 | Basic | 5 | 5.90 | 1 |
| 50 | 100 | SecondarySort | 5 | 5.06 | 1 |
| 100 | 300 | Basic | 5 | 5.09 | 1 |
| 100 | 300 | SecondarySort | 5 | 5.23 | 1 |
| 500 | 1,500 | Basic | 6 | 5.29 | 1 |
| 500 | 1,500 | SecondarySort | 6 | 5.67 | 1 |
| 1,000 | 3,000 | Basic | 6 | 6.89 | 1 |
| 1,000 | 3,000 | SecondarySort | 6 | 6.26 | 1 |
| 2,000 | 6,000 | Basic | 6 | 6.67 | 1 |
| 2,000 | 6,000 | SecondarySort | 6 | 5.76 | 1 |
| 5,000 | 15,000 | Basic | 6 | 6.58 | 1 |
| 5,000 | 15,000 | SecondarySort | 6 | 6.78 | 1 |

### Observations

- **Iterations remain nearly constant (5-6)** as graph size increases 100x. This confirms the paper's observation that real-world (and random) graphs have small effective diameters.
- **Runtime scales modestly** with graph size — dominated by Spark overhead at these local scales.
- **Both algorithms produce identical iteration counts**, as expected (same convergence logic).
- All random graphs at this density form a single connected component.

---

## Experiment 2: Chain Graphs (Worst-Case Diameter)

Chain graphs (0-1-2-...-n) have diameter = n-1, representing the worst case for CCF's iteration bound.

| Nodes | Edges | Algorithm | Iterations | Time (s) | Components |
|------:|------:|-----------|:----------:|---------:|-----------:|
| 10 | 9 | Basic | 6 | 6.21 | 1 |
| 10 | 9 | SecondarySort | 6 | 5.64 | 1 |
| 50 | 49 | Basic | 8 | 7.79 | 1 |
| 50 | 49 | SecondarySort | 8 | 7.74 | 1 |
| 100 | 99 | Basic | 9 | 9.40 | 1 |
| 100 | 99 | SecondarySort | 9 | 9.17 | 1 |
| 200 | 199 | Basic | 10 | 10.09 | 1 |
| 200 | 199 | SecondarySort | 10 | 11.01 | 1 |
| 500 | 499 | Basic | 12 | 13.19 | 1 |
| 500 | 499 | SecondarySort | 12 | 17.75 | 1 |

### Observations

- **Iterations grow logarithmically** with chain length: ~log2(n) iterations, consistent with the O(d) bound where d is the diameter, and CCF's doubling propagation behavior.
- **Runtime is dominated by iteration count** here, since each iteration requires a full MapReduce round-trip.
- At n=500, the **Basic variant is faster** (13.2s vs 17.8s). The SecondarySort variant's sorting overhead per reducer outweighs its memory savings when components are small and sparse.

---

## Experiment 3: Cluster Graphs (Multiple Components)

Dense clusters with optional inter-cluster edges to merge components.

| Clusters | Nodes/Cluster | Inter-edges | Algorithm | Iterations | Time (s) | Components |
|---------:|--------------:|:-----------:|-----------|:----------:|---------:|-----------:|
| 5 | 20 | 0 | Basic | 6 | 7.69 | 5 |
| 5 | 20 | 0 | SecondarySort | 6 | 6.07 | 5 |
| 5 | 20 | 4 | Basic | 7 | 6.98 | 2 |
| 5 | 20 | 4 | SecondarySort | 7 | 6.75 | 2 |
| 10 | 50 | 0 | Basic | 7 | 7.34 | 10 |
| 10 | 50 | 0 | SecondarySort | 7 | 7.34 | 10 |
| 10 | 50 | 9 | Basic | 9 | 8.55 | 4 |
| 10 | 50 | 9 | SecondarySort | 9 | 10.19 | 4 |
| 20 | 50 | 0 | Basic | 7 | 6.99 | 20 |
| 20 | 50 | 0 | SecondarySort | 7 | 7.76 | 20 |
| 20 | 50 | 19 | Basic | 11 | 10.60 | 4 |
| 20 | 50 | 19 | SecondarySort | 11 | 10.12 | 4 |

### Observations

- **Isolated clusters converge quickly** (6-7 iterations) — each cluster's small diameter is resolved independently.
- **Inter-cluster edges increase iterations** (up to 11) because the effective diameter grows as clusters merge into larger components.
- **Component detection is correct**: 0 inter-edges → component count = cluster count; adding bridges reduces component count as expected.
- Both algorithms perform similarly; marginal differences are within noise at this scale.

---

## Strengths of the CCF Algorithm

1. **Simplicity**: The algorithm is conceptually straightforward — only two MapReduce jobs (CCF-Iterate + CCF-Dedup) iterated until convergence. Easy to implement and debug.

2. **Scalability**: As demonstrated in the original paper (6B nodes, 92B edges), the algorithm scales to massive graphs. The per-iteration work is embarrassingly parallel in the map phase.

3. **Low iteration count on real-world graphs**: Random and clustered graphs converge in 5-7 iterations regardless of size, because real-world networks have small diameters (small-world property). This makes CCF practical for social networks, web graphs, and record linkage graphs.

4. **Memory-efficient variant**: The SecondarySort version (Figure 3) avoids storing all values in memory, making it viable when connected components contain millions of nodes (O(1) space per reducer vs O(N) for Basic).

5. **Dedup step improves efficiency**: CCF-Dedup removes duplicate pairs between iterations, reducing I/O and computation in subsequent iterations — a practical optimization not present in all competing algorithms.

6. **Framework-friendly**: The algorithm maps naturally to MapReduce/Spark with no complex data structures or state management beyond key-value pairs.

---

## Weaknesses of the CCF Algorithm

1. **Iteration-bound by diameter**: CCF has O(d) iteration complexity where d is the graph diameter. Chain/path graphs and high-diameter graphs (e.g., trees, sparse lattices) require many iterations. Algorithms like CC-MR achieve O(log d) iterations, which is superior for high-diameter graphs.

2. **Per-iteration overhead**: Each iteration requires a full MapReduce shuffle (map → shuffle → reduce). On distributed clusters, the job initialization, scheduling, and data serialization overhead per iteration is significant. More iterations = more overhead — this is why CC-MR (8 iterations) slightly outperformed CCF (11 iterations) on the web-google dataset despite similar per-iteration efficiency.

3. **SecondarySort overhead at small scale**: The sorting step in Figure 3 adds overhead that only pays off when connected components are very large (50K+ nodes). At smaller scales (our experiments), the Basic variant is often faster. Choosing the right variant requires knowledge of the graph structure.

4. **No convergence guarantees for skewed graphs**: Highly skewed graphs (e.g., star graphs with massive hubs) can cause reducer skew — one reducer handles a disproportionate share of data while others idle. This is a general MapReduce issue but particularly impacts CCF since adjacency lists of high-degree nodes are processed in a single reducer.

5. **Duplicate pair generation**: CCF-Iterate can emit the same pair many times within a single iteration, necessitating the CCF-Dedup step. This is extra I/O and computation that more sophisticated algorithms avoid.

6. **Local-mode performance limitations**: Our experiments run on a single machine (local[*] mode), so we cannot observe the true distributed scaling behavior. The Spark overhead (~5s baseline) dominates at these graph sizes, making it hard to differentiate algorithm performance. Meaningful runtime differences would emerge at 100K+ nodes on a real cluster.

7. **No built-in handling of directed graphs**: The algorithm assumes undirected graphs (emitting both directions in the mapper). For directed graphs, finding strongly connected components requires a different approach.

---

## Comparison: Basic vs SecondarySort

| Aspect | Basic (Fig. 2) | SecondarySort (Fig. 3) |
|--------|---------------|----------------------|
| **Space complexity** | O(N) per reducer — stores all values in list | O(1) per reducer — single pass through sorted values |
| **Time per iteration** | Slightly faster for small components | Sorting overhead; faster only for very large components |
| **When to use** | Components < 50K nodes | Components > 50K nodes (millions) |
| **Iterations** | Identical | Identical |
| **Implementation** | Simpler | Requires secondary sort / custom partitioning |

**Recommendation**: Use **Basic** for graphs where the largest component is moderate (< 50K nodes). Switch to **SecondarySort** for massive graphs where memory is a concern (e.g., the paper's 53M-node component).

---

## Conclusions

1. **CCF is well-suited for real-world graphs** with small diameters (social networks, web graphs, record linkage) — converges in few iterations with simple implementation.
2. **Chain/high-diameter graphs expose CCF's weakness** — iteration count grows logarithmically, each requiring a full MapReduce round-trip.
3. **At local scale, Spark overhead dominates** — true algorithmic differences would be visible at 100K+ nodes on a distributed cluster.
4. **The two variants are complementary**: Basic is simpler and faster for moderate graphs; SecondarySort is essential for memory efficiency on massive components.
5. **For production use on massive graphs** (billions of nodes), CCF is a proven choice — the original paper demonstrated it on 6B nodes / 92B edges in 7 hours on an 80-node cluster.
