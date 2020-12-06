package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 11:33
 * Desc 演示特征工程-特征提取-TFIDF
 */
object C_1_TFIDFDemo {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("sparkml_tfidf").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    val data: DataFrame = spark.createDataFrame(
      Seq((0, "Hi I heard about Spark Spark Spark Spark Spark Spark Spark"),
        (0, "I wish Java could use case classes"),
        (1, "Logistic regression models are neat")))
      .toDF("label", "words")

    //1.分词
    val token:Tokenizer = new Tokenizer()
      .setInputCol("words")//指定对哪一列进行分词
      .setOutputCol("token_words")//指定分完之后的列名
    val tokenWords: DataFrame = token.transform(data)
    tokenWords.show(false)

    //2.准备TF
    val hashTF: HashingTF = new HashingTF()
      .setInputCol("token_words") //指定要对哪一列的数据进行计算TF,肯定是要指定分好词的哪一列
      .setOutputCol("tf_words")
    val tf_DataFrame: DataFrame = hashTF.transform(tokenWords)

    //3.准备IDF
    val idf: IDF = new IDF()
      .setInputCol("tf_words") //指定要对哪一列的数据进行计算IDF
      .setOutputCol("idf_words")
    val model: IDFModel = idf.fit(tf_DataFrame)

    //4.真正的计算结果
    val result: DataFrame = model.transform(tf_DataFrame)
    result.show(false)

    //对象.transform:将一个DataFrame按照对象的规则转换为另一个DataFrame
    //对象.fit:适合/填充/训练得到一个model,然后后续可以使用model.transform将一个DataFrame按照model转换为另一个DataFrame

  }
}
