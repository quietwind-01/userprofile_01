package cn.itcast.up.mldemo

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * Author itcast
 * Date 2020/3/20 10:34
 * Desc 演示SparkMLlib中统计学相关API
 */
object B_StaticDemo {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("model")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    val rdd: RDD[Array[Double]] = sc.parallelize(List(Array(1.0),Array(2.0),Array(3.0),Array(4.0),Array(5.0)))
    val vectorRDD: RDD[linalg.Vector] = rdd.map(arr=>Vectors.dense(arr))

    println("=======1.查看基本统计学特征=======")
    //1.查看基本统计学特征
    //MultivariateStatisticalSummary:多元统计摘要
    val summary: MultivariateStatisticalSummary = Statistics.colStats(vectorRDD)
    println(summary.max)
    println(summary.min)
    println(summary.mean)
    println(summary.variance)//方差=sum((元素-均值)^2) / n   标准差= 根号下方差
    println(summary.normL1)//L1范数:元素绝对值的和
    println(summary.normL2)//L2范数:元素绝对值的平方和再开方
    //[5.0]
    //[1.0]
    //[3.0]
    //[2.4999999999999996]
    //[15.0]
    //[7.416198487095663]

    println("=======2.查看统计学相关性=======")
    //2.查看统计学相关性
    val x: RDD[Double] = sc.parallelize(Array(1,2,3,4,5))
    val y: RDD[Double] = sc.parallelize(Array(10,20,30,40,50))
    val z: RDD[Double] = sc.parallelize(Array(-10,-20,-30,-40,-50))
    //y = 10x ===> y和x正相关===>完全的正相关相关系数为1
    //z = -10x===> z和x负相关===>完全的负相关相关系数为-1
    val corr1: Double = Statistics.corr(x,y)//1.0,默认为pearson相关系数
    val corr2: Double = Statistics.corr(x,z)//-1.0
    val corr3: Double = Statistics.corr(x,y,"spearman")//0.9999999999999998
    println(corr1)
    println(corr2)
    println(corr3)
    //https://blog.csdn.net/lambsnow/article/details/79972145
 }
}
