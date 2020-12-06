package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 15:14
 * Desc 演示特征工程-特征转换-VectorAssembler向量装配器
 * 把多个向量合并为一个向量
 * [1], [2] ===> [1,2]
 *
 */
object D_2_VectorAssembler {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("sparkml").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    val data: DataFrame = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")
    data.show(false)
    /*
 +---+----+------+--------------+-------+
|id |hour|mobile|userFeatures  |clicked|
+---+----+------+--------------+-------+
|0  |18  |1.0   |[0.0,10.0,0.5]|1.0    |
+---+----+------+--------------+-------+
目标是将众多特征合并为一个特征向量如:
[18 ,1.0,  0.0,10.0,0.5,   1.0]

     */

    //1.创建VectorAssembler向量装配器
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures", "clicked"))
      .setOutputCol("vectorAssember_feature")
    val result: DataFrame = vectorAssembler.transform(data)

    result.show(false)
    /*
 +---+----+------+--------------+-------+---------------------------+
|id |hour|mobile|userFeatures  |clicked|vectorAssember_feature     |
+---+----+------+--------------+-------+---------------------------+
|0  |18  |1.0   |[0.0,10.0,0.5]|1.0    |[18.0,1.0,0.0,10.0,0.5,1.0]|
+---+----+------+--------------+-------+---------------------------+
     */
  }
}
