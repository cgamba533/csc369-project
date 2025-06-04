import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.math.pow

object KNNModelRed {
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

      // Exclude residual sugar (3) and pH (8)
      val selectedIndices = (0 until 11).filterNot(i => i == 3 || i == 8)
      val selectedFeatures = selectedIndices.map(features(_)).toArray

      (selectedFeatures, quality)
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
    val k = 16
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
    ///////// Join predictions with actual labels to calculate errors
    val actuals = testRDD.map { case (features, quality) => (features.toVector, quality) }

    val joined = predictions.join(actuals) // (features) => (predicted, actual)

    val mae = joined.map { case (_, (pred, actual)) =>
      math.abs(pred - actual)
    }.mean()

    val mse = joined.map { case (_, (pred, actual)) =>
      math.pow(pred - actual, 2)
    }.mean()

    println(f"\nEvaluation Metrics:")
    println(f"Mean Absolute Error (MAE): $mae%.4f")
    println(f"Mean Squared Error (MSE): $mse%.4f")//////////

    // Compute R² (coefficient of determination)
    val meanActual = joined.map { case (_, (_, actual)) => actual }.mean()

    val ssTotal = joined.map { case (_, (_, actual)) =>
      math.pow(actual - meanActual, 2)
    }.sum()

    val ssResidual = joined.map { case (_, (pred, actual)) =>
      math.pow(pred - actual, 2)
    }.sum()

    val r2 = 1 - (ssResidual / ssTotal)

    // Compute RMSE (Root Mean Squared Error)
    val rmse = math.sqrt(mse)
    println(f"Root Mean Squared Error (RMSE): $rmse%.4f")


    val predictionAndLabels = predictions.join(actuals).map {
      case (_, (predicted, actual)) =>
        val predLabel = math.round(predicted).toInt
        val trueLabel = math.round(actual).toInt
        (predLabel, trueLabel)
    }

    val threshold = 7

    val binaryLabels = predictionAndLabels.map { case (pred, label) =>
      val predBinary = if (pred >= threshold) 1 else 0
      val labelBinary = if (label >= threshold) 1 else 0
      (predBinary, labelBinary)
    }
    val metrics = binaryLabels.map {
        case (pred, actual) =>
          if (pred == 1 && actual == 1) ("TP", 1)
          else if (pred == 1 && actual == 0) ("FP", 1)
          else if (pred == 0 && actual == 1) ("FN", 1)
          else if (pred == 0 && actual == 0) ("TN", 1)
          else ("Other", 0)
      }
      .reduceByKey(_ + _)
      .collectAsMap()

    val tp = metrics.getOrElse("TP", 0)
    val fp = metrics.getOrElse("FP", 0)
    val fn = metrics.getOrElse("FN", 0)
    val tn = metrics.getOrElse("TN", 0)
    val total = tp + fp + fn + tn

    val accuracy = (tp + tn).toDouble / total
    val precision = if (tp + fp > 0) tp.toDouble / (tp + fp) else 0.0
    val recall = if (tp + fn > 0) tp.toDouble / (tp + fn) else 0.0
    val f1 = if (precision + recall > 0) 2 * (precision * recall) / (precision + recall) else 0.0

    println("\nClassification Metrics (threshold ≥ 7):")
    println(f"Accuracy : $accuracy%.4f")



    sc.stop()
  }
}