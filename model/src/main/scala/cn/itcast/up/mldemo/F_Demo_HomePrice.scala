package cn.itcast.up.mldemo

import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

/**
 * Author itcast
 * Date 2020/3/22 9:35
 * Desc 演示使用SparkMLlib-API对房价数据进行特征工程处理,并使用线性回归做房价预测
 * 注意:API使用基于RDD的和基于DataFrame的都可以,
 * 但是学习测试时为了方便演示底层细节,这里选用基于RDD的
 * 后面项目的模型标签中会使用基于DataFrame
 */
object F_Demo_HomePrice {

  //准备样例类封装数据
  case class Home(
                   mlsNum: Double, //mlsNum: Double,
                   city: String, //城市city: String,
                   sqFt: Double, //平方英尺sqFt: Double,
                   bedrooms: Double, //卧室数bedrooms: Double,
                   bathrooms: Double, //卫生间数bathrooms: Double,
                   garage: Double, //车库garage: Double,
                   age: Double, //房屋年龄age: Double,
                   acres: Double, //房屋占地面积 acres: Double,
                   price: Double //房屋价格price: Double
                 )

  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val conf: SparkConf = new SparkConf().setMaster("local[2]").setAppName("HomePrice")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val fileRDD: RDD[String] = sc.textFile("file:///D:\\授课\\191021-35\\用户画像\\day07\\data\\ml\\data\\homeprice.data")
    //fileRDD.foreach(println)
    /*
4424038|Farmington|2058.0|4|2|3|11|0.21|249900.0
4415730|Burnsville|2419.0|4|3|2|52|0.572|249900.0
4432734|Burnsville|2592.0|4|3|2|40|0.279|249900.0
4433641|Burnsville|2984.0|4|3|2|47|0.26|249900.0
4418340|Eagan|2300.0|4|2|2|26|0.246|249900.0
4407113|Burnsville|2708.0|3|3|3|39|0.33|249900.0
4403311|Burnsville|2130.0|4|3|2|18|0.34|249900.0
4412250|Burnsville|2024.0|3|3|2|27|0.04|249000.0
4419748|Lakeville|2336.0|3|2|2|36|0.428|248900.0
     */
    val homeRDD: RDD[Home] = fileRDD.map(line => {
      val arr: Array[String] = line.split("[|]")
      Home(
        arr(0).toDouble,
        arr(1).toString,
        arr(2).toDouble,
        arr(3).toDouble,
        arr(4).toDouble,
        arr(5).toDouble,
        arr(6).toDouble,
        arr(7).toDouble,
        arr(8).toDouble
      )
    })
    //homeRDD.take(5).foreach(println)
    /*
Home(4424109.0,Apple Valley,1634.0,2.0,2.0,2.0,33.0,0.04,119900.0)
Home(4404211.0,Rosemount,13837.0,4.0,6.0,4.0,17.0,14.46,3500000.0)
Home(4339082.0,Burnsville,9040.0,4.0,6.0,8.0,12.0,0.74,2690000.0)
Home(4362154.0,Lakeville,6114.0,7.0,5.0,12.0,25.0,14.83,1649000.0)
Home(4388419.0,Lakeville,6546.0,5.0,5.0,11.0,38.0,5.28,1575000.0)
     */

    //1.数据向量化--先将价格向量化
    val priceVector: RDD[linalg.Vector] = homeRDD.map(home => {
      Vectors.dense(home.price)
    })

    //2.查看价格的统计学特征(也可以查看其他列)
    val summary: MultivariateStatisticalSummary = Statistics.colStats(priceVector)
    println("max:" + summary.max)
    println("min:" + summary.min)
    println("mean:" + summary.mean)
    println("variance:" + summary.variance)
    /*
max:[3500000.0]
min:[1.0]
mean:[295117.888888889]----μ
variance:[4.738079624862584E10]----方差---σ^2
取值几乎全部集中在（μ-3σ,μ+3σ)区间内，超出这个范围的可能性仅占不到0.3%.
取值在区间(μ-3σ,μ+3σ)之外的概率很小，是小概率事件，通常认为是不会发生的。
所以价格在这个区间之外的数据可以理解为异常数据,过滤掉
(295117.8-3*2.2,295117.8+3*2.2)
(295111.2,595124.4)
也就是只要数据在(295111.2,595124.4)范围外可以理解为异常值(其他是大量房屋的价格是呈正态分布的)
所以应该将在这个范围的数据进行保留
但是需要注意:如果我们按照这个范围(295111.2,595124.4)认为是正常的数据,那么整体数据只剩下296条
而在这个范围之外的数据有559条
不符合我们的正态分布的规则,正态分布的3σ说的是在这之外的数据只占0.3%.,而现在数据明显不符合
所以不能完全套用
所以应制定其他的异常数据的标准
     */

    //3.过滤掉异常数据--也可以再次统计学特征
    //过滤出的异常数据
    val falseData: RDD[Home] = homeRDD.filter(home => home.price > 100000000 || home.price < 2 || home.sqFt < 2 || home.sqFt > 100000)
    println("falseData")
    println(falseData.count())
    //过滤出正常数据
    val trueData: RDD[Home] = homeRDD.filter(home => (home.price > 2 && home.price < 100000000) || (home.sqFt > 2 && home.sqFt < 100000))
    println("trueData")
    println(trueData.count())
    //实际中数据异常的范围应该要根据实际情况制定过滤规则
    //我们这里没有过多的去观察其他数据,只是人为的做一些简单的规则过滤,比较粗糙
    /*
falseData
1
trueData
855
     */


    //4.查看价格和面积的相关系数(也可查看其他的列与列之间的相关系数)
    val corr: Double = Statistics.corr(trueData.map(home => home.price), trueData.map(home => home.sqFt))
    println(corr) //0.8324532722940541

    //5.将数据转为标签向量LabeledPoint(label就是价格,features就是特征)
    val labeledPointRDD: RDD[LabeledPoint] = trueData.map(home => {
      LabeledPoint(home.price, Vectors.dense(home.age, home.bathrooms, home.bedrooms, home.garage, home.sqFt))
    })
    //val featureRDD: RDD[linalg.Vector] = labeledPointRDD.map(lp=>lp.features)
    //6.对数据进行归一化
    val model: StandardScalerModel = new StandardScaler(true,true).fit(labeledPointRDD.map(lp=>lp.features))
    val scalerRDD: RDD[LabeledPoint] = labeledPointRDD.map(lp => {
      LabeledPoint(lp.label, model.transform(lp.features))
    })
    scalerRDD.take(5).foreach(println)
    /*
(119900.0,[0.6680193403026262,-0.816651014752078,-1.4782047359828068,-0.4103042909566379,-0.718742804337014])
(3500000.0,[-0.23544131964200302,3.321047459991785,0.48127596055253796,1.7960489717347152,10.003314500643391])
(2690000.0,[-0.5177727758746996,3.321047459991785,0.48127596055253796,6.208755497117422,5.788473159006827])
(1649000.0,[0.2162890103303116,2.286622841305819,3.4204970053555552,10.621462022500127,3.2175693095949875])
(1575000.0,[0.9503507965353228,2.286622841305819,1.4610163088202104,9.518285391154452,3.597142263438431])
     */

    //7.扩展:使用线性回归训练模型
    val linearRegression: LinearRegressionWithSGD = new LinearRegressionWithSGD()
      .setIntercept(true)//添加截距,类似y=ax + b  要b
    linearRegression
      .optimizer
      .setNumIterations(1000)//SGD随机梯度下降需要指定最大迭代次数
      .setStepSize(0.1)//指定随机梯度下降的步长
    //划分数据集
    val Array(testSet,trainSet)= scalerRDD.randomSplit(Array(0.3,0.7),100)
    //训练模型
    val linearRegressionModel: LinearRegressionModel = linearRegression.run(trainSet)

    //8.扩展:评价模型
    //sum((真实价格-预测价格)^2)/count也就是均方误差
    val priceTuple: RDD[(Double, Double)] = trainSet.map(lp => {
      val predictPrice: Double = linearRegressionModel.predict(lp.features)
      //返回(真实价格,预测价格)
      (lp.label, predictPrice)
    })
    val powerSum: Double = priceTuple.map(t => {
      //(真实价格-预测价格)^2
      math.pow(t._1 - t._2, 2)
    }).reduce(_ + _)
    //mse:Mean Square Error 均方误差
    val mse = powerSum / priceTuple.count()
    println("在训练集上的均方误差为:"+mse)//越小越好
    //在训练集上的均方误差为:1.0885809856019258E10

    //9.扩展:使用模型对测试集做预测
    //sum((真实价格-预测价格)^2)/count也就是均方误差
    val priceTuple2: RDD[(Double, Double)] = testSet.map(lp => {
      val predictPrice: Double = linearRegressionModel.predict(lp.features)
      //返回(真实价格,预测价格)
      (lp.label, predictPrice)
    })
    val powerSum2: Double = priceTuple2.map(t => {
      //(真实价格-预测价格)^2
      math.pow(t._1 - t._2, 2)
    }).reduce(_ + _)
    //mse:Mean Square Error 均方误差
    val mse2 = powerSum2 / priceTuple2.count()
    println("在测试集上的均方误差为:"+mse2)//越小越好
    //在测试集上的均方误差为:4.159244783940943E10

    priceTuple2.take(5).foreach(println) //查看模型对测试集的预测
    //真实价格,预测价格
    //(3500000.0,1004300.9712813771)
    //(2690000.0,1078207.6867769503)
    //(975000.0,465720.28435131756)
    //(975000.0,929715.1395072498)
    //(778500.0,678232.1683192339)

  }
}
