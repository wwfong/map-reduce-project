/**
 * CCF: Connected Component Finder using MapReduce (Spark)
 * ========================================================
 * Scala implementation of the CCF algorithm from:
 * "CCF: Fast and Scalable Connected Component Computation in MapReduce"
 * by Kardes, Agrawal, Wang, and Sun (inome Inc.)
 *
 * Implements:
 *   - Figure 2: CCF-Iterate (basic version)
 *   - Figure 3: CCF-Iterate with secondary sorting
 *   - Figure 4: CCF-Dedup
 */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator

object CCFConnectedComponents {

  // ===========================================================================
  // Figure 2: CCF-Iterate (Basic Version)
  // ===========================================================================

  /**
   * CCF-Iterate (Figure 2): Generates adjacency lists for each node,
   * finds minimum neighbor, and emits new pairs to propagate component IDs.
   *
   * Pseudocode:
   *   map(key, value):
   *     emit(key, value)
   *     emit(value, key)
   *
   *   reduce(key, <iterable> values):
   *     min = key
   *     for each (value in values):
   *       if value < min: min = value
   *       valueList.add(value)
   *     if min < key:
   *       emit(key, min)
   *       for each (value in valueList):
   *         if min != value:
   *           Counter.NewPair.increment(1)
   *           emit(value, min)
   */
  def ccfIterate(
      pairsRDD: RDD[(String, String)],
      newPairCounter: LongAccumulator
  ): RDD[(String, String)] = {

    // Map phase: emit both directions
    val mapped: RDD[(String, String)] = pairsRDD.flatMap { case (key, value) =>
      Seq((key, value), (value, key))
    }

    // Reduce phase: group by key, find min, emit new pairs
    val reduced: RDD[(String, String)] = mapped.groupByKey().flatMap {
      case (key, valuesIter) =>
        val valueList = valuesIter.toList
        var min = key
        for (value <- valueList) {
          if (value < min) min = value
        }

        if (min < key) {
          // Emit (key, min)
          val results = scala.collection.mutable.ListBuffer((key, min))
          // Emit (value, min) for all other values
          for (value <- valueList) {
            if (value != min) {
              newPairCounter.add(1)
              results += ((value, min))
            }
          }
          results.toList
        } else {
          List.empty[(String, String)]
        }
    }

    reduced
  }

  // ===========================================================================
  // Figure 3: CCF-Iterate with Secondary Sorting
  // ===========================================================================

  /**
   * CCF-Iterate with secondary sorting (Figure 3): More memory-efficient
   * version using sorted values so the first value is always the minimum.
   *
   * Pseudocode:
   *   map(key, value):
   *     emit(key, value)
   *     emit(value, key)
   *
   *   reduce(key, <iterable> values):  [values arrive sorted]
   *     minValue = values.next()
   *     if minValue < key:
   *       emit(key, minValue)
   *       for each (value in values):
   *         Counter.NewPair.increment(1)
   *         emit(value, minValue)
   */
  def ccfIterateSecondarySort(
      pairsRDD: RDD[(String, String)],
      newPairCounter: LongAccumulator
  ): RDD[(String, String)] = {

    // Map phase: emit both directions
    val mapped: RDD[(String, String)] = pairsRDD.flatMap { case (key, value) =>
      Seq((key, value), (value, key))
    }

    // Reduce phase with secondary sorting
    val reduced: RDD[(String, String)] = mapped.groupByKey().flatMap {
      case (key, valuesIter) =>
        val sortedValues = valuesIter.toList.sorted

        if (sortedValues.isEmpty) {
          List.empty[(String, String)]
        } else {
          val minValue = sortedValues.head

          if (minValue < key) {
            // Emit (key, minValue)
            val results = scala.collection.mutable.ListBuffer((key, minValue))
            // Emit (value, minValue) for remaining values
            for (value <- sortedValues.tail) {
              newPairCounter.add(1)
              results += ((value, minValue))
            }
            results.toList
          } else {
            List.empty[(String, String)]
          }
        }
    }

    reduced
  }

  // ===========================================================================
  // Figure 4: CCF-Dedup
  // ===========================================================================

  /**
   * CCF-Dedup (Figure 4): Deduplicates the output of CCF-Iterate.
   *
   * Pseudocode:
   *   map(key, value):
   *     temp.entity1 = key
   *     temp.entity2 = value
   *     emit(temp, null)
   *
   *   reduce(key, <iterable> values):
   *     emit(key.entity1, key.entity2)
   */
  def ccfDedup(pairsRDD: RDD[(String, String)]): RDD[(String, String)] = {
    // The paper uses composite key (entity1, entity2) -> null for dedup
    // In Spark, this maps to:
    pairsRDD
      .map(pair => (pair, null))    // map: emit((key,value), null)
      .reduceByKey((a, _) => a)     // reduce: group by composite key
      .map(_._1)                    // emit(key.entity1, key.entity2)
  }

  // ===========================================================================
  // Full CCF Pipeline
  // ===========================================================================

  /**
   * Full CCF pipeline: iteratively runs CCF-Iterate + CCF-Dedup until
   * no new pairs are generated.
   *
   * @param sc             SparkContext
   * @param edges          List of (nodeA, nodeB) edges
   * @param useSecondarySort  If true, use the secondary sort version (Figure 3)
   * @param maxIterations  Maximum number of iterations
   * @return RDD of (node, componentId) pairs
   */
  def findConnectedComponents(
      sc: SparkContext,
      edges: Seq[(String, String)],
      useSecondarySort: Boolean = false,
      maxIterations: Int = 100
  ): RDD[(String, String)] = {

    var pairsRDD: RDD[(String, String)] = sc.parallelize(edges)
    var iteration = 0
    var converged = false

    while (iteration < maxIterations && !converged) {
      iteration += 1
      println(s"--- CCF Iteration $iteration ---")

      // Reset counter for this iteration
      val newPairCounter: LongAccumulator = sc.longAccumulator("NewPairCounter")

      // CCF-Iterate (Figure 2 or Figure 3)
      val iterateOutput: RDD[(String, String)] = if (useSecondarySort) {
        ccfIterateSecondarySort(pairsRDD, newPairCounter)
      } else {
        ccfIterate(pairsRDD, newPairCounter)
      }

      // CCF-Dedup (Figure 4)
      val dedupedRDD = ccfDedup(iterateOutput)
      dedupedRDD.cache()

      // Force evaluation to trigger accumulators
      val pairCount = dedupedRDD.count()
      val newPairs = newPairCounter.value

      println(s"  Pairs after dedup: $pairCount, New pairs: $newPairs")

      // Unpersist old RDD
      pairsRDD.unpersist()
      pairsRDD = dedupedRDD

      // Check convergence
      if (newPairs == 0) {
        println(s"Converged after $iteration iterations!")
        converged = true
      }
    }

    pairsRDD
  }

  // ===========================================================================
  // Main - Example Driver
  // ===========================================================================

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("CCF-ConnectedComponents")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    // Example graph from the paper (Figure 5):
    //   A--B, B--D, D--E, A--C, A--E, F--G, F--H
    //
    // Expected components:
    //   Component 1: {A, B, C, D, E} -> component ID = A
    //   Component 2: {F, G, H}       -> component ID = F

    val edges = Seq(
      ("A", "B"),
      ("B", "D"),
      ("D", "E"),
      ("A", "C"),
      ("A", "E"),
      ("F", "G"),
      ("F", "H")
    )

    // --- Basic Version (Figure 2) ---
    println("=" * 60)
    println("CCF Connected Components - Basic Version (Figure 2)")
    println("=" * 60)

    val components = findConnectedComponents(sc, edges, useSecondarySort = false)
    println("\nNode -> Component ID mapping:")
    components.collect().sorted.foreach { case (node, compId) =>
      println(s"  $node -> $compId")
    }

    // --- Secondary Sort Version (Figure 3) ---
    println()
    println("=" * 60)
    println("CCF Connected Components - Secondary Sort (Figure 3)")
    println("=" * 60)

    val componentsSS = findConnectedComponents(sc, edges, useSecondarySort = true)
    println("\nNode -> Component ID mapping:")
    componentsSS.collect().sorted.foreach { case (node, compId) =>
      println(s"  $node -> $compId")
    }

    // --- Summary ---
    println()
    println("=" * 60)
    println("Connected Components Summary")
    println("=" * 60)

    val compGroups = components
      .map { case (node, compId) => (compId, node) }
      .groupByKey()
      .mapValues(_.toList)
      .collect()
      .sorted

    compGroups.foreach { case (compId, members) =>
      println(s"  Component $compId: ${(members :+ compId).distinct.sorted.mkString(", ")}")
    }

    sc.stop()
  }
}
