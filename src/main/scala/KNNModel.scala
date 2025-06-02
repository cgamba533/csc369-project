import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.math.pow

object KNNModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("KNNModel").setMaster("local[4]")
    val sc = new SparkContext(conf)

    // Importing Dataset
    val dataRaw = sc.textFile("winequality-red.csv")
    val heading = dataRaw.first()

    val data = dataRaw.filter(_ != heading)
      .map(_.split(";"))
      .filter(_.length >= 12)

    val parsed = data.map(fields => {
      val features = fields.take(11).map(_.toDouble)
      val quality = fields(11).toDouble
      (features, quality)
    })

    // Normalize Data
    val numFeatures = parsed.first()._1.length
    val mins = parsed.map(_._1).reduce((a, b) => a.zip(b).map { case (x, y) => math.min(x, y) })
    val maxs = parsed.map(_._1).reduce((a, b) => a.zip(b).map {case (x, y) => math.max(x,y) })

    val normData = parsed.map { case (features, quality) =>
      val normFeatures = features.zipWithIndex.map { case (value, i) =>
        if (maxs(i) == mins(i)) 0.0 else (value - mins(i)) / (maxs(i) - mins(i))
      }
      (normFeatures, quality)
    }

    // Train-test split
    val dataLength = normData.count().toInt
    val trainSize = (dataLength * 0.9).toInt

    val sortedData = normData.zipWithIndex().map(_.swap).sortByKey().map(_._2)
    val trainRDD = sortedData.zipWithIndex().filter(_._2 < trainSize).map(_._1).persist()
    val testRDD = sortedData.zipWithIndex().filter(_._2 >= trainSize).map(_._1)

    // KNN (K = 10)
    val k = 10
    val cartesian = testRDD.cartesian(trainRDD)

    val distances = cartesian.map { case ((testFeatures, _), (trainFeatures, trainQuality)) =>
      val dist = math.sqrt(testFeatures.zip(trainFeatures).map { case (a, b) => pow (a - b, 2)}.sum)
      (testFeatures.toVector, (dist, trainQuality))
    }

    val grouped = distances.groupByKey()

    val predictions = grouped.mapValues( iter => {
      val nearest = iter.toList.sortBy(_._1).take(k)
      val avgQuality = nearest.map(_._2).sum / k
      avgQuality
    })

    // Show Predictions
    println("\nSample Predictions:")
    predictions.collect().foreach { case (features, predQuality) => // .take(10) for sample predictions
      println(f"Predicted Quality: $predQuality%.2f | Features: ${features.mkString(", ")}")
    }

    sc.stop()
  }
}
