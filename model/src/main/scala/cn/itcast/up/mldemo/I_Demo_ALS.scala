package cn.itcast.up.mldemo

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Author itcast
 * Date 2020/3/28 11:31
 * Desc 演示使用SparkMLlib提供的基于隐语义模型的协同过滤推荐算法-ALS
 */
object I_Demo_ALS {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val sparkConf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("ALS")
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("WARN")
    val data: RDD[String] = sc.textFile("file:///D:\\授课\\191021-35\\用户画像\\day07\\data\\ml\\recommend\\u.data")
    data.take(5).foreach(println)
    /*
用户id 电影id  评分 时间(不要)
196	    242	    3	  881250949
186	    302	    3	  891717742
22	    377	    1	  878887116
244	    51	    2	  880606923
166	    346	    1	  886397596
     */

    //1.数据处理-将上面的数据转为ALS需要的数据格式:Rating评分对象
    import org.apache.spark.mllib.recommendation.Rating
    val ratingArrRDD: RDD[Array[String]] = data.map(_.split("\t").take(3))//每行取前3列
    val ratingRDD: RDD[Rating] = ratingArrRDD.map(arr => {
      // Rating (user: Int,product: Int,rating: Double)
      Rating(arr(0).toInt, arr(1).toInt, arr(2).toDouble)
    })

    //2.训练模型--可以理解为矩阵分解
    //ratings 用户/物品/评分数据组成的RDD,传入后可以转为评分矩阵
    //rank    暂时可以理解为K
    //iterations 迭代次数(分解矩阵的时候用的是交替最小二乘法,需要指定迭代次数)
    //lambda    迭代步长(分解矩阵的时候用的是交替最小二乘法,需要指定迭代步长)
    val model = ALS.train(ratingRDD,10,20,0.1)

    //3.预测
    //预测196号用户对242号电影的评分
    val result: Double = model.predict(196,242)
    println("预测196号用户对242号电影的评分为:"+result)
    //给196号用户推荐5个电影/商品/音乐/新闻...
    val movies: Array[Rating] = model.recommendProducts(196,5)
    println("给196号用户推荐5个电影:"+movies.toBuffer)
    //给346号电影推荐5个用户
    val users: Array[Rating] = model.recommendUsers(346,5)
    println("给346号电影推荐5个用户:"+users.toBuffer)

    /*
预测196号用户对242号电影的评分为:3.7492196188012796
给196号用户推荐5个电影:ArrayBuffer(
Rating(196,1643,5.641637658464943),
Rating(196,1463,5.002882946480057),
Rating(196,1131,4.75816744212836),
Rating(196,1512,4.743447980575427),
Rating(196,1449,4.728815671746528))
给346号电影推荐5个用户:ArrayBuffer(
Rating(118,346,4.789554936277204),
Rating(252,346,4.72952398889646),
Rating(628,346,4.680364224189548),
Rating(928,346,4.633902595712135),
Rating(519,346,4.607606396189652))
     */

  }
}
