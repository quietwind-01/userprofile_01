package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/20 15:14
 * Desc 演示特征工程-特征转换-StandarScaler-标准缩放器/定标器
 */
object D_3_StandarScaler {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder().master("local[*]").appName("sparkml_tfidf").getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    val path = "file:///D:\\data\\spark\\ml\\data\\sample_libsvm_data.txt"
    val data: DataFrame = spark.read.format("libsvm").load(path)
    data.show(false)
    data.printSchema()
    /*
 +-----+-------------------------
|label|features
+-----+-------------------------
|0.0  |(692,[127,128,129,130,131
|1.0  |(692,[158,159,160,161,185
|1.0  |(692,[124,125,126,127,151
|1.0  |(692,[152,153,154,180,181
|1.0  |(692,[151,152,153,154,179
|0.0  |(692,[129,130,131,132,156
|1.0  |(692,[158,159,160,185,186
|1.0  |(692,[99,100,101,127,128,
     */

    //1.使用StandarScaler对数据进行标准缩放
    val standardScaler: StandardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("StandardScaler_features")
      .setWithMean(false)//默认false表示不以原数据的均值居中,而以0为均值
      .setWithStd(true)//默认true表示是否将数据缩放为单位标准偏差1
    val model: StandardScalerModel = standardScaler.fit(data)
    val result: DataFrame = model.transform(data)
    result.select("StandardScaler_features")show(false)
    /*
     */


    //问题:问什么要对数据进行standardScaler标准化缩放
    //举例,如研究身高,体重,年龄...等对人体健康的影响,收集到数据如下
    //180cm,70kg,29岁
    //和
    //1.8m,140斤,29岁
    //上面的两条数据表示的含义应该是一样的,但是值不一样,最后结果肯定受影响,所以需要统一单位,如统一为:
    //1.8m,70kg,29岁
    //统一到上面的形式ok吗?
    //答案:不行,在真正进行运行的时候,因为m的数据范围为1~3m,而kg的数据范围为:30~300
    //那么数据的取值范围差异较大,那么训练出来的模型可能模型参数就过大
    //如 y = a1x1身高 + a2x2体重 + a3x3,中a2参数对结果影响较大,因为a2参数变化一点点,就会被x2(30~300)放大对y的影响
    //所以在实际开发中为了消除量纲(单位)对结果的影响,需要对数据进行归一化处理,
    //而归一化处理的常用方式就是standardScaler标准化缩放(或其他如最小最大归一化MinMaxScaler)
    //standardScaler标准化缩放就是将数据全部缩放到标准正态分布
    //标准正态分布是位置参数u =0，尺度参数sigma^2 = 1的正态分布
    //那么对于每一个元素进行标准化缩放之后的值 = (原数据 - 列的均值) /标准差


  }
}
