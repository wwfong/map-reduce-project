/**
 * CCF Experiments: Benchmarking on synthetic graphs
 * ==================================================
 * Mirrors ccf_experiments.py — runs Basic and SecondarySort variants
 * on random, chain, and cluster graphs; records iterations, runtime, components.
 *
 * Run with:
 *   spark-submit --class CCFExperiments CCFExperiments.jar
 * or via spark-shell / spark-submit with --packages
 */

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
  // CCF Algorithm (from CCFConnectedComponents.scala)
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
    val csvPath = "/Users/davidwfong/Code/Master/map-reduce-project/experiment_results_scala.csv"
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
