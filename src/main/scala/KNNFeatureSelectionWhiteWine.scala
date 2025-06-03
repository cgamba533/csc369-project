import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.math.pow

object KNNFeatureSelectionWhiteWine {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("KNNFeatureSelection").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val dataRaw = sc.textFile("winequality-white.csv") // Change to white if needed
    val heading = dataRaw.first()

    val data = dataRaw.filter(_ != heading)
      .map(_.split(";"))
      .filter(_.length >= 12)

    val parsed = data.map(fields => {
      val features = fields.take(11).map(_.toDouble)
      val quality = fields(11).toDouble
      (features, quality)
    }).cache()

    val numFeatures = parsed.first()._1.length
    val mins = parsed.map(_._1).reduce((a, b) => a.zip(b).map { case (x, y) => math.min(x, y) })
    val maxs = parsed.map(_._1).reduce((a, b) => a.zip(b).map { case (x, y) => math.max(x, y) })

    def normalize(data: org.apache.spark.rdd.RDD[(Array[Double], Double)], includedIndices: Array[Int]) = {
      data.map { case (features, quality) =>
        val normFeatures = includedIndices.map { i =>
          val value = features(i)
          if (maxs(i) == mins(i)) 0.0 else (value - mins(i)) / (maxs(i) - mins(i))
        }
        (normFeatures, quality)
      }
    }

    def runKNN(data: org.apache.spark.rdd.RDD[(Array[Double], Double)], k: Int): Double = {
      val dataLength = data.count().toInt
      val trainSize = (dataLength * 0.9).toInt

      val sortedData = data.zipWithIndex().map(_.swap).sortByKey().map(_._2)
      val trainRDD = sortedData.zipWithIndex().filter(_._2 < trainSize).map(_._1).persist()
      val testRDD = sortedData.zipWithIndex().filter(_._2 >= trainSize).map(_._1)

      val cartesian = testRDD.cartesian(trainRDD)

      val distances = cartesian.map { case ((testFeatures, _), (trainFeatures, trainQuality)) =>
        val dist = math.sqrt(testFeatures.zip(trainFeatures).map { case (a, b) => pow(a - b, 2) }.sum)
        (testFeatures.toVector, (dist, trainQuality))
      }

      val grouped = distances.groupByKey()

      val predictions = grouped.mapValues(iter => {
        val nearest = iter.toList.sortBy(_._1).take(k)
        val avgQuality = nearest.map(_._2).sum / k
        avgQuality
      })

      val trueLabels = testRDD.map { case (features, quality) => (features.toVector, quality) }
      val joined = predictions.join(trueLabels)

      val mse = joined.map { case (_, (pred, actual)) => pow(pred - actual, 2) }.mean()
      math.sqrt(mse) // RMSE
    }

    val k = 10
    println(s"\nEvaluating feature importance using Leave-One-Feature-Out:")

    // Run full model first
    val fullIndices = (0 until numFeatures).toArray
    val normFull = normalize(parsed, fullIndices)
    val fullRMSE = runKNN(normFull, k)
    println(f"All features RMSE: $fullRMSE%.4f")

    // LOFO analysis
    (0 until numFeatures).foreach { dropIdx =>
      val keptIndices = fullIndices.filter(_ != dropIdx)
      val normData = normalize(parsed, keptIndices)
      val rmse = runKNN(normData, k)
      println(f"Dropping feature $dropIdx%2d => RMSE: $rmse%.4f (↑ ${rmse - fullRMSE}%.4f)")
    }

    sc.stop()
  }
}
