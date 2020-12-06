package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 16:30
 * Desc 演示特征工程-特征选择-卡方选择器(卡方检验)
 */
object E_ChiSqSelector {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("sparkml_tfidf").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    val seq: Seq[(Int, linalg.Vector, Double)] = Seq(
      //gender: 0–女 1–男
      //hobby: 1-篮球 2-足球 3-乒乓 4-逛街 5-购物
      //age：年龄
      //salary：工资
      //makeup：是否化妆 1–化妆 0–不化妆
      //id, gender,hobby,age,salary, makeup
      (1, Vectors.dense(0.0, 4, 22.0, 4000), 1.0),
      (2, Vectors.dense(0.0, 4, 22.0, 6000), 1.0),
      (3, Vectors.dense(0.0, 5, 25.0, 5000), 1.0),
      (4, Vectors.dense(0.0, 5, 28.0, 7000), 1.0),
      (5, Vectors.dense(0.0, 3, 25.0, 1000), 1.0),
      (6, Vectors.dense(0.0, 4, 30.0, 5000), 0.0),
      (7, Vectors.dense(1.0, 1, 21.0, 6000), 0.0),
      (8, Vectors.dense(1.0, 1, 22.0, 4000), 0.0),
      (9, Vectors.dense(1.0, 3, 23.0, 6000), 0.0),
      (9, Vectors.dense(1.0, 2, 25.0, 7000), 0.0)
    )
    val data = spark.createDataFrame(seq).toDF("id", "features", "label")
    data.show(false)
    /*
+---+---------------------+-----+
|id |features             |label|
+---+---------------------+-----+
|1  |[0.0,4.0,22.0,4000.0]|1.0  |
|2  |[0.0,4.0,22.0,6000.0]|1.0  |
|3  |[0.0,5.0,25.0,5000.0]|1.0  |
|4  |[0.0,5.0,28.0,7000.0]|1.0  |
|5  |[0.0,3.0,25.0,1000.0]|1.0  |
|6  |[0.0,4.0,30.0,5000.0]|0.0  |
|7  |[1.0,1.0,21.0,6000.0]|0.0  |
|8  |[1.0,1.0,22.0,4000.0]|0.0  |
|9  |[1.0,3.0,23.0,6000.0]|0.0  |
|9  |[1.0,2.0,25.0,7000.0]|0.0  |
+---+---------------------+-----+
     */

    //1.使用卡方选择器进行特征选择
   val selector: ChiSqSelector =  new ChiSqSelector()
      .setFeaturesCol("features")//设置特征列
      .setLabelCol("label")//设置标签列
      .setOutputCol("selectedFeature")//设置被选择的特征列列名
      .setNumTopFeatures(1)//设置选择和结果列/标签列最为相关的几个特征

    //训练/适合/填充
    val model: ChiSqSelectorModel = selector.fit(data)

    //转换/预测
    val result: DataFrame = model.transform(data)
    result.show(false)
    /*
+---+---------------------+-----+---------------+
|id |features             |label|selectedFeature|
+---+---------------------+-----+---------------+
|1  |[0.0,4.0,22.0,4000.0]|1.0  |[0.0]          |
|2  |[0.0,4.0,22.0,6000.0]|1.0  |[0.0]          |
|3  |[0.0,5.0,25.0,5000.0]|1.0  |[0.0]          |
|4  |[0.0,5.0,28.0,7000.0]|1.0  |[0.0]          |
|5  |[0.0,3.0,25.0,1000.0]|1.0  |[0.0]          |
|6  |[0.0,4.0,30.0,5000.0]|0.0  |[0.0]          |
|7  |[1.0,1.0,21.0,6000.0]|0.0  |[1.0]          |
|8  |[1.0,1.0,22.0,4000.0]|0.0  |[1.0]          |
|9  |[1.0,3.0,23.0,6000.0]|0.0  |[1.0]          |
|9  |[1.0,2.0,25.0,7000.0]|0.0  |[1.0]          |
+---+---------------------+-----+---------------+
     */

  }
}
