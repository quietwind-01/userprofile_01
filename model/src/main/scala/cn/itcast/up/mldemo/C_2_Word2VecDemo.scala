package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 11:50
 * Desc 演示特征工程-特征提取-Word2Vec
 * 把word用一个低维稠密向量表示,解决one-hot编码造成的维数灾难和语义相关问题。
 */
object C_2_Word2VecDemo {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder()
      .appName("SparkMlilb")
      .master("local[*]")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")

    val data: DataFrame = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")
    data.show(false)
    /*
+------------------------------------------+
|text                                      |
+------------------------------------------+
|[Hi, I, heard, about, Spark]              |
|[I, wish, Java, could, use, case, classes]|
|[Logistic, regression, models, are, neat] |
+------------------------------------------+
     */

    //2.使用word2Vec
    val word2Vec: Word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model: Word2VecModel = word2Vec.fit(data)
    val result: DataFrame = model.transform(data)
    result.show(false)
    /*
+------------------------------------------+----------------------------------------------------------------+
|text                                      |result                                                          |
+------------------------------------------+----------------------------------------------------------------+
|[Hi, I, heard, about, Spark]              |[-0.008142343163490296,0.02051363289356232,0.03255096450448036] |
|[I, wish, Java, could, use, case, classes]|[0.043090314205203734,0.035048123182994974,0.023512658663094044]|
|[Logistic, regression, models, are, neat] |[0.038572299480438235,-0.03250147425569594,-0.01552378609776497]|
+------------------------------------------+----------------------------------------------------------------+
     */
  }
}
