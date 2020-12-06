package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{IndexToString, OneHotEncoder, StringIndexer, StringIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 14:33
 * Desc 演示特征工程-特征转换-StringIndexer_IndexToString_OneHotEncoder
 */
object D_1_StringIndexer_IndexToString_OneHotEncoder {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("sparkml_tfidf").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    val data = spark.createDataFrame(
      Seq((0, "红"), (1, "绿"), (2, "蓝"))).
      toDF("id", "words")
    data.show()
    /*
+---+-----+
| id|words|
+---+-----+
|  0|    红|
|  1|    绿|
|  2|    蓝|
+---+-----+
     */


    //SparkMLlib中后续的计算都是基于向量/矩阵
    //但是实际中数据的特征有很多是字符串,所以需要对字符串进行编码
    //1.StringIndexer将字符串编码为数字,如红橙黄绿青蓝紫编码为1234567
    val stringIndexer: StringIndexer = new StringIndexer()
      .setInputCol("words")
      .setOutputCol("index_words")
    val stringIndexerModel: StringIndexerModel = stringIndexer.fit(data)
    val indexResult: DataFrame = stringIndexerModel.transform(data)
    indexResult.show(false)
    /*
+---+-----+-----------+
|id |words|index_words|
+---+-----+-----------+
|0  |红    |1.0        |
|1  |绿    |0.0        |
|2  |蓝    |2.0        |
+---+-----+-----------+
     */


    //2.IndexToString将数字解码为字符串,如1234567解码为红橙黄绿青蓝紫
    val indexToString: IndexToString = new IndexToString()
      .setInputCol("index_words")
      .setOutputCol("before_words")
      .setLabels(stringIndexerModel.labels)
    val beforeResult: DataFrame = indexToString.transform(indexResult)
    beforeResult.show(false)
    /*
+---+-----+-----------+------------+
|id |words|index_words|before_words|
+---+-----+-----------+------------+
|0  |红    |1.0        |红           |
|1  |绿    |0.0        |绿           |
|2  |蓝    |2.0        |蓝           |
+---+-----+-----------+------------+
     */


    //3.OneHotEncoder独热编码
    //上面的编码红橙黄绿青蓝紫编码为1234567,那么在计算的时候可能会出现2>1的情况,但是实际业务中并不想表达橙>红
    //所以就不应该使用StringIndexer,而应该使用别的方法,如OneHotEncoder独热编码
    //如:对红绿蓝进行编码:
    //+-----+-------+-------------+
    //|红    |1.0   |(2,[1],[1.0])|2,11
    //|绿    |0.0   |(2,[0],[1.0])|2,01
    //|蓝    |2.0   |(2,[],[])    |2,00
    //+-----+-------+-------------+
    //注意:OneHotEncoder的input必须为Index数字类型,所以需要先进行StringIndexer
    val oneHotEncoder: OneHotEncoder = new OneHotEncoder()
      .setInputCol("index_words")
      .setOutputCol("oneHot_Result")
    val oneHotResult: DataFrame = oneHotEncoder.transform(indexResult)
    oneHotResult.show(false)
    /*
+---+-----+-----------+-------------+
|id |words|index_words|oneHot_Result|
+---+-----+-----------+-------------+
|0  |红    |1.0        |(2,[1],[1.0])|11
|1  |绿    |0.0        |(2,[0],[1.0])|01
|2  |蓝    |2.0        |(2,[],[])    |00
+---+-----+-----------+-------------+
     */

  }
}
