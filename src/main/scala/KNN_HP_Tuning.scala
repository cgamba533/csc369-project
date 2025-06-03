import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.math.pow

object KNN_HP_Tuning {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("KNN_HP_Tuning").setMaster("local[4]")
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

      val selectedIndices = (0 until 11).filterNot(i => i == 3 || i == 8)
      val selectedFeatures = selectedIndices.map(features(_)).toArray

      (selectedFeatures, quality)
    })

    // Normalize Data
    val numFeatures = parsed.first()._1.length
    val mins = parsed.map(_._1).reduce((a, b) => a.zip(b).map { case (x, y) => math.min(x, y) })
    val maxs = parsed.map(_._1).reduce((a, b) => a.zip(b).map { case (x, y) => math.max(x, y) })

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
      .zipWithIndex()
      .map { case ((features, label), i) => (i, (features, label)) }

    // Hyperparameter tuning
    val kValues = (1 to 100).toList
    val results = kValues.map { k =>
      val cartesian = testRDD.mapValues(_._1).cartesian(trainRDD)

      val distances = cartesian.map { case ((testIdx, testFeatures), (trainFeatures, trainQuality)) =>
        val dist = math.sqrt(testFeatures.zip(trainFeatures).map { case (a, b) => pow(a - b, 2) }.sum)
        (testIdx, (dist, trainQuality))
      }


      val predictions = distances.groupByKey().mapValues { iter =>
        val kNearest = iter.toList.sortBy(_._1).take(k)
        val avg = kNearest.map(_._2).sum / k
        avg
      }
      // Test Predictions
      val joinedPred = predictions.join(testRDD)
      val squaredError = joinedPred.map { case (_, (pred, (_, actual))) =>
        pow(pred - actual, 2)
      }

      val rmse = math.sqrt(squaredError.mean())
      println(f"k = $k => RMSE: $rmse%.4f")
      (k, rmse)
    }

    val best = results.minBy(_._2)
    println(f"\nBest K val: ${best._1} => RMSE: ${best._2}%.4f")
  }
}
