package cn.itcast.up.mldemo

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 14:24
 * Desc 演示特征工程-特征提取-CountVector
 * 做单词计算(只统计单词在该篇文章中的次数)
 */
object C_3_CountVectorDemo {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("sparkml_tfidf").getOrCreate()
    val data: DataFrame = spark.createDataFrame(
      Seq((0, Array("a", "b", "c")),
        (1, Array("a", "b", "b", "c", "a")))).
      toDF("id", "words")
    //1.创建CountVectorizer
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("count_words")
    val model: CountVectorizerModel = countVectorizer.fit(data)
    val result: DataFrame = model.transform(data)
    //2.查看结果
    result.show(false)
    /*
+---+---------------+-------------------------+
|id |words          |count_words              |
+---+---------------+-------------------------+
|0  |[a, b, c]      |(3,[0,1,2],[1.0,1.0,1.0])|
|1  |[a, b, b, c, a]|(3,[0,1,2],[2.0,2.0,1.0])|
+---+---------------+-------------------------+
     */

  }
}
