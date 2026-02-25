"""
CCF Experiments: Benchmarking CCF algorithms on synthetic graphs
================================================================
Generates synthetic graphs of increasing size and topology, runs both
CCF-Iterate variants, and records iterations and runtime.
"""

import time
import random
import csv
from pyspark import SparkContext, SparkConf


# =============================================================================
# Graph Generators
# =============================================================================

def generate_chain_graph(n):
    """Chain: 0-1-2-...-n-1. Diameter = n-1 (worst case for iterations)."""
    return [(str(i), str(i + 1)) for i in range(n - 1)]


def generate_random_graph(n_nodes, n_edges, seed=42):
    """Erdos-Renyi style random graph with fixed number of edges."""
    rng = random.Random(seed)
    edges = set()
    while len(edges) < n_edges:
        a = rng.randint(0, n_nodes - 1)
        b = rng.randint(0, n_nodes - 1)
        if a != b:
            edge = (str(min(a, b)), str(max(a, b)))
            edges.add(edge)
    return list(edges)


def generate_cluster_graph(n_clusters, nodes_per_cluster, inter_edges=0, seed=42):
    """Multiple dense clusters with optional inter-cluster edges."""
    rng = random.Random(seed)
    edges = []
    # Intra-cluster: connect each node to ~3 neighbors within cluster
    for c in range(n_clusters):
        base = c * nodes_per_cluster
        for i in range(nodes_per_cluster - 1):
            edges.append((str(base + i), str(base + i + 1)))
            # Add a few random intra-cluster edges for density
            if i + 2 < nodes_per_cluster:
                edges.append((str(base + i), str(base + i + 2)))
    # Inter-cluster edges
    for _ in range(inter_edges):
        c1, c2 = rng.sample(range(n_clusters), 2)
        n1 = c1 * nodes_per_cluster + rng.randint(0, nodes_per_cluster - 1)
        n2 = c2 * nodes_per_cluster + rng.randint(0, nodes_per_cluster - 1)
        edges.append((str(n1), str(n2)))
    return edges


# =============================================================================
# CCF Implementations (inlined for experiment)
# =============================================================================

def ccf_iterate_basic(pairs_rdd):
    mapped = pairs_rdd.flatMap(lambda pair: [(pair[0], pair[1]), (pair[1], pair[0])])
    grouped = mapped.groupByKey()

    def reduce_fn(kv):
        key, values_iter = kv
        value_list = list(values_iter)
        min_val = key
        for v in value_list:
            if v < min_val:
                min_val = v
        results = []
        new_pairs = 0
        if min_val < key:
            results.append((key, min_val))
            for v in value_list:
                if v != min_val:
                    new_pairs += 1
                    results.append((v, min_val))
        return results, new_pairs

    reduced = grouped.map(reduce_fn).cache()
    output_rdd = reduced.flatMap(lambda x: x[0])
    new_pair_count = reduced.map(lambda x: x[1]).reduce(lambda a, b: a + b)
    reduced.unpersist()
    return output_rdd, new_pair_count


def ccf_iterate_secondary_sort(pairs_rdd):
    mapped = pairs_rdd.flatMap(lambda pair: [(pair[0], pair[1]), (pair[1], pair[0])])
    grouped = mapped.groupByKey().mapValues(lambda vals: sorted(vals))

    def reduce_fn(kv):
        key, sorted_values = kv
        results = []
        new_pairs = 0
        if not sorted_values:
            return results, new_pairs
        min_value = sorted_values[0]
        if min_value < key:
            results.append((key, min_value))
            for v in sorted_values[1:]:
                new_pairs += 1
                results.append((v, min_value))
        return results, new_pairs

    reduced = grouped.map(reduce_fn).cache()
    output_rdd = reduced.flatMap(lambda x: x[0])
    new_pair_count = reduced.map(lambda x: x[1]).reduce(lambda a, b: a + b)
    reduced.unpersist()
    return output_rdd, new_pair_count


def ccf_dedup(pairs_rdd):
    return pairs_rdd.distinct()


def run_ccf(sc, edges, iterate_fn, max_iterations=100):
    """Run full CCF pipeline, return (iterations, runtime_seconds, num_components)."""
    start = time.time()
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

    elapsed = time.time() - start

    # Count components
    num_components = pairs_rdd.map(lambda x: x[1]).distinct().count()

    return iteration, elapsed, num_components


# =============================================================================
# Experiment Runner
# =============================================================================

def main():
    conf = SparkConf().setAppName("CCF-Experiments").setMaster("local[*]")
    conf.set("spark.ui.showConsoleProgress", "false")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    results = []

    # ---- Experiment 1: Increasing graph size (random graphs) ----
    print("=" * 70)
    print("EXPERIMENT 1: Random graphs of increasing size")
    print("=" * 70)
    print(f"{'Nodes':>8} {'Edges':>8} {'Algorithm':>15} {'Iters':>6} {'Time(s)':>10} {'Components':>12}")
    print("-" * 70)

    random_configs = [
        (50,    100),
        (100,   300),
        (500,   1500),
        (1000,  3000),
        (2000,  6000),
        (5000,  15000),
    ]

    for n_nodes, n_edges in random_configs:
        edges = generate_random_graph(n_nodes, n_edges)

        for name, fn in [("Basic", ccf_iterate_basic), ("SecondarySort", ccf_iterate_secondary_sort)]:
            iters, elapsed, n_comp = run_ccf(sc, edges, fn)
            print(f"{n_nodes:>8} {n_edges:>8} {name:>15} {iters:>6} {elapsed:>10.3f} {n_comp:>12}")
            results.append({
                "experiment": "random_graph",
                "nodes": n_nodes,
                "edges": n_edges,
                "algorithm": name,
                "iterations": iters,
                "runtime_sec": round(elapsed, 3),
                "components": n_comp,
            })

    # ---- Experiment 2: Chain graphs (worst case diameter) ----
    print()
    print("=" * 70)
    print("EXPERIMENT 2: Chain graphs (worst-case diameter)")
    print("=" * 70)
    print(f"{'Nodes':>8} {'Edges':>8} {'Algorithm':>15} {'Iters':>6} {'Time(s)':>10} {'Components':>12}")
    print("-" * 70)

    chain_sizes = [10, 50, 100, 200, 500]

    for n in chain_sizes:
        edges = generate_chain_graph(n)

        for name, fn in [("Basic", ccf_iterate_basic), ("SecondarySort", ccf_iterate_secondary_sort)]:
            iters, elapsed, n_comp = run_ccf(sc, edges, fn)
            print(f"{n:>8} {n-1:>8} {name:>15} {iters:>6} {elapsed:>10.3f} {n_comp:>12}")
            results.append({
                "experiment": "chain_graph",
                "nodes": n,
                "edges": n - 1,
                "algorithm": name,
                "iterations": iters,
                "runtime_sec": round(elapsed, 3),
                "components": n_comp,
            })

    # ---- Experiment 3: Cluster graphs ----
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Cluster graphs (multiple components)")
    print("=" * 70)
    print(f"{'Clusters':>9} {'Nodes/C':>8} {'Inter':>6} {'Algorithm':>15} {'Iters':>6} {'Time(s)':>10} {'Components':>12}")
    print("-" * 70)

    cluster_configs = [
        (5,   20,  0),    # 5 isolated clusters
        (5,   20,  4),    # 5 clusters, some connected
        (10,  50,  0),    # 10 isolated clusters
        (10,  50,  9),    # 10 clusters, all connected
        (20,  50,  0),    # 20 isolated clusters
        (20,  50,  19),   # 20 clusters, all connected
    ]

    for n_clusters, npc, inter in cluster_configs:
        edges = generate_cluster_graph(n_clusters, npc, inter)
        total_nodes = n_clusters * npc

        for name, fn in [("Basic", ccf_iterate_basic), ("SecondarySort", ccf_iterate_secondary_sort)]:
            iters, elapsed, n_comp = run_ccf(sc, edges, fn)
            print(f"{n_clusters:>9} {npc:>8} {inter:>6} {name:>15} {iters:>6} {elapsed:>10.3f} {n_comp:>12}")
            results.append({
                "experiment": "cluster_graph",
                "nodes": total_nodes,
                "edges": len(edges),
                "algorithm": name,
                "iterations": iters,
                "runtime_sec": round(elapsed, 3),
                "components": n_comp,
                "clusters": n_clusters,
                "inter_edges": inter,
            })

    # ---- Save results to CSV ----
    csv_path = "/Users/davidwfong/Code/Master/map-reduce-project/experiment_results.csv"
    fieldnames = ["experiment", "nodes", "edges", "algorithm", "iterations",
                  "runtime_sec", "components", "clusters", "inter_edges"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults saved to {csv_path}")

    sc.stop()


if __name__ == "__main__":
    main()
