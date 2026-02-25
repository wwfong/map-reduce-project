"""
CCF: Connected Component Finder using MapReduce
================================================
Python implementation of the CCF algorithm from:
"CCF: Fast and Scalable Connected Component Computation in MapReduce"
by Kardes, Agrawal, Wang, and Sun (inome Inc.)

This implements:
- Figure 2: CCF-Iterate (basic version)
- Figure 3: CCF-Iterate with secondary sorting
- Figure 4: CCF-Dedup

Uses PySpark for the MapReduce framework.
"""

from pyspark import SparkContext, SparkConf


# =============================================================================
# Figure 2: CCF-Iterate (Basic Version)
# =============================================================================

def ccf_iterate(pairs_rdd):
    """
    CCF-Iterate (Figure 2): Generates adjacency lists for each node,
    finds minimum neighbor, and emits new pairs to propagate component IDs.

    Pseudocode from paper:
        map(key, value):
            emit(key, value)
            emit(value, key)

        reduce(key, <iterable> values):
            min = key
            for each (value in values):
                if value < min:
                    min = value
                valueList.add(value)
            if min < key:
                emit(key, min)
                for each (value in valueList):
                    if min != value:
                        Counter.NewPair.increment(1)
                        emit(value, min)

    Args:
        pairs_rdd: RDD of (node_a, node_b) edge pairs

    Returns:
        (output_rdd, new_pair_count): Tuple of output RDD and count of new pairs
    """
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
            # Emit (key, min)
            results.append((key, min_val))
            # Emit (value, min) for all other values
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


# =============================================================================
# Figure 3: CCF-Iterate with Secondary Sorting
# =============================================================================

def ccf_iterate_secondary_sort(pairs_rdd):
    """
    CCF-Iterate with secondary sorting (Figure 3): More memory-efficient
    version that uses sorted values to avoid storing all values in a list.

    Pseudocode from paper:
        map(key, value):
            emit(key, value)
            emit(value, key)

        reduce(key, <iterable> values):  [values arrive sorted]
            minValue = values.next()
            if minValue < key:
                emit(key, minValue)
                for each (value in values):
                    Counter.NewPair.increment(1)
                    emit(value, minValue)

    Args:
        pairs_rdd: RDD of (node_a, node_b) edge pairs

    Returns:
        (output_rdd, new_pair_count): Tuple of output RDD and count of new pairs
    """
    # Map phase: emit both directions
    mapped = pairs_rdd.flatMap(lambda pair: [
        (pair[0], pair[1]),
        (pair[1], pair[0])
    ])

    # Reduce phase with secondary sorting:
    # Sort values so first value is the minimum
    grouped = mapped.groupByKey().mapValues(lambda vals: sorted(vals))

    def reduce_fn(key_values):
        key, sorted_values = key_values
        results = []
        new_pairs = 0

        if not sorted_values:
            return results, new_pairs

        min_value = sorted_values[0]

        if min_value < key:
            # Emit (key, minValue)
            results.append((key, min_value))
            # Emit (value, minValue) for remaining values
            for value in sorted_values[1:]:
                new_pairs += 1
                results.append((value, min_value))

        return results, new_pairs

    reduced = grouped.map(reduce_fn).cache()
    output_rdd = reduced.flatMap(lambda x: x[0])
    new_pair_count = reduced.map(lambda x: x[1]).reduce(lambda a, b: a + b)

    reduced.unpersist()
    return output_rdd, new_pair_count


# =============================================================================
# Figure 4: CCF-Dedup
# =============================================================================

def ccf_dedup(pairs_rdd):
    """
    CCF-Dedup (Figure 4): Deduplicates the output of CCF-Iterate.

    Pseudocode from paper:
        map(key, value):
            temp.entity1 = key
            temp.entity2 = value
            emit(temp, null)

        reduce(key, <iterable> values):
            emit(key.entity1, key.entity2)

    Args:
        pairs_rdd: RDD of (node_a, node_b) pairs (may contain duplicates)

    Returns:
        Deduplicated RDD of (node_a, node_b) pairs
    """
    # In Spark, this is simply distinct()
    # The paper uses the composite key (entity1, entity2) -> null pattern
    # to leverage MapReduce's built-in grouping for dedup
    return pairs_rdd.distinct()


# =============================================================================
# Full CCF Pipeline
# =============================================================================

def find_connected_components(sc, edges, use_secondary_sort=False, max_iterations=100):
    """
    Full CCF pipeline: iteratively runs CCF-Iterate + CCF-Dedup until
    no new pairs are generated.

    Args:
        sc: SparkContext
        edges: List of (node_a, node_b) tuples representing edges
        use_secondary_sort: If True, use the secondary sort version (Figure 3)
        max_iterations: Maximum number of iterations

    Returns:
        RDD of (node, component_id) pairs where component_id is the
        smallest node ID in each connected component
    """
    iterate_fn = ccf_iterate_secondary_sort if use_secondary_sort else ccf_iterate

    pairs_rdd = sc.parallelize(edges)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"--- CCF Iteration {iteration} ---")

        # CCF-Iterate
        output_rdd, new_pair_count = iterate_fn(pairs_rdd)

        # CCF-Dedup
        pairs_rdd = ccf_dedup(output_rdd)
        pairs_rdd.cache()

        pair_count = pairs_rdd.count()
        print(f"  Pairs after dedup: {pair_count}, New pairs: {new_pair_count}")

        # If no new pairs were generated, we've converged
        if new_pair_count == 0:
            print(f"Converged after {iteration} iterations!")
            break

    # The final pairs_rdd contains (node, component_id) mappings
    return pairs_rdd


# =============================================================================
# Example / Driver
# =============================================================================

if __name__ == "__main__":
    conf = SparkConf().setAppName("CCF-ConnectedComponents").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Example graph from the paper (Figure 5):
    #   A--B, B--D, D--E, A--C, A--E, F--G, F--H
    #
    # Expected components:
    #   Component 1: {A, B, C, D, E} -> component ID = A
    #   Component 2: {F, G, H}       -> component ID = F

    edges = [
        ("A", "B"),
        ("B", "D"),
        ("D", "E"),
        ("A", "C"),
        ("A", "E"),
        ("F", "G"),
        ("F", "H"),
    ]

    print("=" * 60)
    print("CCF Connected Components - Basic Version (Figure 2)")
    print("=" * 60)
    components = find_connected_components(sc, edges, use_secondary_sort=False)
    print("\nNode -> Component ID mapping:")
    for node, comp_id in sorted(components.collect()):
        print(f"  {node} -> {comp_id}")

    print()
    print("=" * 60)
    print("CCF Connected Components - Secondary Sort (Figure 3)")
    print("=" * 60)
    components_ss = find_connected_components(sc, edges, use_secondary_sort=True)
    print("\nNode -> Component ID mapping:")
    for node, comp_id in sorted(components_ss.collect()):
        print(f"  {node} -> {comp_id}")

    # Summarize connected components
    print("\n" + "=" * 60)
    print("Connected Components Summary")
    print("=" * 60)
    comp_groups = components.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)
    for comp_id, members in sorted(comp_groups.collect()):
        print(f"  Component {comp_id}: {sorted(members + [comp_id])}")

    sc.stop()
