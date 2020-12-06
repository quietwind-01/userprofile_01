package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.common.HDFSUtils
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg
import org.apache.spark.sql._

import scala.collection.immutable

/**
 * Author itcast
 * Date 2020/3/22 16:16
 * Desc 使用KMeans算法根据用户的RFM的值计算客户价值模型+实现模型的保存和加载
 *
 */
object RFMModel2 extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }

  //37	客户价值		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_orders##family=detail##selectFields=memberId,orderSn,orderAmount,finishTime		4	36
  override def getTagId(): Long = 37

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    hbaseDF.show(false)
    hbaseDF.printSchema()
    fiveDS.show(false)
    fiveDS.printSchema()
/*
用户id      订单编号            订单金额    完成时间
+---------+-------------------+-----------+----------+
|memberId |orderSn            |orderAmount|finishTime|
+---------+-------------------+-----------+----------+
|4035167 |ts_792756751164275 |2479.45     |1564415022|
|4035167  |D14090106121770839 |2449.00    |1565687310|
|4035291  |D14090112394810659 |1099.42    |1564681801|
|4035041  |fx_787749561729045 |1999.00    |1565799378|

root
 |-- memberId: string (nullable = true)
 |-- orderSn: string (nullable = true)
 |-- orderAmount: string (nullable = true)
 |-- finishTime: string (nullable = true)

+---+----+
|id |rule|
+---+----+
|38 |1   |超高价值
|39 |2   |高价值
|40 |3   |中上价值
|41 |4   |中价值
|42 |5   |中下价值
|43 |6   |低价值
|44 |7   |超低价值
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
 */
    //0.方便后续API的调用,导入隐式转换
    import org.apache.spark.sql.functions._
    import spark.implicits._

    //0.防止后续的字符串拼写错误,可以定义一些常量
    val recencyStr: String = "recency"
    val frequencyStr: String  = "frequency"
    val monetaryStr: String  = "monetary"
    val featureStr: String  = "feature"
    val predictStr: String  = "predict"

    //1.通过以上的数据我们就可以求出每个用的RFM:按照用户id分组后求:
    //R:Recency最近一次消费时间距离今天的天数(取分组后用户的最近/最大的订单时间和今天的时间求天数差)
    //F:Frequency最近一段时间消费次数(直接取分组后该用户的全部订单数)
    //M:Monetary最近一段时间消费总金额(直接取分组后该用户的全部订单总金额)
    //1.1声明Column对象表示RFM如何计算
    //max('finishTime)求用户的最近/最大的订单时间
    //from_unixtime将数据转为时间对象
    //current_date求当前时间
    //date_sub(current_date(),200)把当前时间往前推200天
    //datediff求两个时间相差的天数
    val recencyColumn:Column =datediff(date_sub(current_date(),206), from_unixtime(max('finishTime))) as recencyStr
    val frequencyColumn:Column = count('orderSn) as frequencyStr
    val monetaryColumn:Column = sum('orderAmount) as monetaryStr

    //1.2按照用户id分组求RFM
    val RFMDF: DataFrame = hbaseDF.groupBy('memberId as "userId")
      .agg(recencyColumn, frequencyColumn, monetaryColumn)
    RFMDF.show(false)
    /*
 +---------+-------+---------+------------------+
|userId   |recency|frequency|monetary          |
+---------+-------+---------+------------------+
|13822725 |12     |116      |179298.34         |
|13823083 |12     |132      |233524.17         |
|138230919|12     |125      |240061.56999999998|
|13823681 |12     |108      |169746.1          |
     */

    //2.数据归一化/标准化(可以使用SparkMLlib的归一化工具,也可以使用运营/产品提供的打分规则(更具有业务含义),或打完分之后再使用SparkMLlib的归一化工具再做一次)
    //R: 1-3天=5分，4-6天=4分，7-9天=3分，10-15天=2分，大于16天=1分
    //F: ≥200=5分，150-199=4分，100-149=3分，50-99=2分，1-49=1分
    //M: ≥20w=5分，10-19w=4分，5-9w=3分，1-4w=2分，<1w=1分
    val recencyScore: Column = functions.when(col(recencyStr) >= 1 && col(recencyStr) <= 3,5)
      .when(col(recencyStr) >= 4 && col(recencyStr) <= 6,4)
      .when((col(recencyStr) >= 7) && (col(recencyStr) <= 9), 3)
      .when((col(recencyStr) >= 10) && (col(recencyStr) <= 15), 2)
      .when(col(recencyStr) >= 16, 1)
      .as(recencyStr)
    val frequencyScore: Column = functions.when(col(frequencyStr) >= 200, 5)
      .when((col(frequencyStr) >= 150) && (col(frequencyStr) <= 199), 4)
      .when((col(frequencyStr) >= 100) && (col(frequencyStr) <= 149), 3)
      .when((col(frequencyStr) >= 50) && (col(frequencyStr) <= 99), 2)
      .when((col(frequencyStr) >= 1) && (col(frequencyStr) <= 49), 1)
      .as(frequencyStr)
    val monetaryScore: Column = functions.when(col(monetaryStr) >= 200000, 5)
      .when(col(monetaryStr).between(100000, 199999), 4)
      .when(col(monetaryStr).between(50000, 99999), 3)
      .when(col(monetaryStr).between(10000, 49999), 2)
      .when(col(monetaryStr) <= 9999, 1)
      .as(monetaryStr)

    val RFMScoreDF: DataFrame = RFMDF.select('userId,recencyScore,frequencyScore,monetaryScore)
    RFMScoreDF.show(false)
    RFMScoreDF.printSchema()
    /*
 +---------+-------+---------+--------+
|userId   |recency|frequency|monetary|
+---------+-------+---------+--------+
|13822725 |2      |3        |4       |
|13823083 |2      |3        |5       |
|138230919|2      |3        |5       |
|13823681 |2      |3        |4       |
     */

    //3.特征向量化
    //使用VectorAssembler可以将多列值装配为一个向量
    val vectorDF: DataFrame = new VectorAssembler()
      .setInputCols(Array(recencyStr, frequencyStr, monetaryStr))
      .setOutputCol(featureStr)
      .transform(RFMScoreDF)
    vectorDF.show(false)
    vectorDF.printSchema()
    /*
  +---------+-------+---------+--------+-------------+
|userId   |recency|frequency|monetary|feature      |
+---------+-------+---------+--------+-------------+
|13822725 |2      |3        |4       |[2.0,3.0,4.0]|
|13823083 |2      |3        |5       |[2.0,3.0,5.0]|
|138230919|2      |3        |5       |[2.0,3.0,5.0]|

root
 |-- userId: string (nullable = true)
 |-- recency: integer (nullable = true)
 |-- frequency: integer (nullable = true)
 |-- monetary: integer (nullable = true)
 |-- feature: vector (nullable = true)
     */
    //=======================4.实现模型的保存和加载============================
    //0.声明模型存储的位置
    val path:String = "/model/RFM35"//hdfs://bd001:8020/model/RFM35
    //0.声明模型
    var model: KMeansModel = null

    //1.判断HDFS上是否有该模型/路径
    if(HDFSUtils.getInstance().exists(path)){//该路径/模型存在,则直接加载并使用
      println("模型存在于HDFS,直接加载并使用")
      model = KMeansModel.load(path)
    }else{//该路径/模型不存在,重新训练并保存
      println("模型不存在,重新训练并保存到HDFS")
      model= new KMeans()
        .setK(7) //暂时先使用7,后续会讲解如何选择
        .setSeed(200) //随机种子
        .setMaxIter(20) //最大迭代次数
        .setFeaturesCol(featureStr)
        .setPredictionCol(predictStr)
        .fit(vectorDF)//训练/填充/适合
      model.save(path)
    }
    //=======================实现模型的保存和加载============================

    //5.预测
    val predictResultDF: DataFrame = model.transform(vectorDF)
    predictResultDF.show(false)
    /*
 +---------+-------+---------+--------+-------------+-------+
|userId   |recency|frequency|monetary|feature      |predict|
+---------+-------+---------+--------+-------------+-------+
|13822725 |2      |3        |4       |[2.0,3.0,4.0]|2      |
|13823083 |2      |3        |5       |[2.0,3.0,5.0]|0      |
|138230919|2      |3        |5       |[2.0,3.0,5.0]|0      |
|13823681 |2      |3        |4       |[2.0,3.0,4.0]|2      |
|4033473  |2      |3        |5       |[2.0,3.0,5.0]|0      |
|13822841 |2      |3        |5       |[2.0,3.0,5.0]|0      |
|13823153 |2      |3        |5       |[2.0,3.0,5.0]|0      |
|13823431 |2      |3        |4       |[2.0,3.0,4.0]|2      |
|4033348  |2      |3        |5       |[2.0,3.0,5.0]|0      |
     */

    //6.进一步查看每个聚类中心的RFM的和的最大最小值,并按照聚类中心索引编号排序
    /*val RFMClusterInfo = predictResultDF
      .groupBy(predictStr)
      .agg(
        max(col(recencyStr) + col(frequencyStr) + col(monetaryStr)),
        min(col(recencyStr) + col(frequencyStr) + col(monetaryStr))
      )
      .sort(col(predictStr))//默认升序
    RFMClusterInfo.show(false)*/
    /*
+-------+--------+--------+
|predict|max(RFM)|min(RFM)|
+-------+-------+---------+
|0      |10     |9        |
|1      |4      |3        |
|2      |9      |9        |
|3      |12     |11       |
|4      |7      |7        |
|5      |6      |5        |
|6      |8      |8        |
+-------+------+---------------------------------------+
     */

    //目标是将预测出的用户的聚类中心索引编号和用户价值等级id对应,也就是返回userId,tagsId
    //7.将聚类中心索引编号和5级标签对应起来
    //通过分析我们知道,聚类中心的RFM的sum越大说明该聚类中心的价值越大,那么根据聚类的原理,
    //聚到该类/簇的其他用户应该和聚类中心是很相近的也是价值较大的
    //那么如果我们可以将聚类中心的RFM的sum按照从大到小排好序,那么就得出了聚类中心的价值大小
    //7.1将聚类中心的RFM的sum按照从大到小排序,得出按sum排序后的[predict聚类中心索引编号,聚类中心的RFM的sum]
    val centers: Array[linalg.Vector] = model.clusterCenters//获取所有聚类中心
    val indices: Range = centers.indices//获取聚类中心的索引编号组成的集合
    //IndexedSeq[(predict聚类中心索引编号,聚类中心的RFM的sum)]
    val indexAndSum: immutable.IndexedSeq[(Int, Double)] = indices.map(index => {
      val center: linalg.Vector = centers(index) //根据索引取聚类中心
      val RFMSum: Double = center.toArray.sum
      (index, RFMSum)
    }).sortBy(_._2).reverse//逆序
    indexAndSum.foreach(println)
    /*
(3,11.038461538461538)
(0,9.998603351955307)
(2,8.966666666666667)
(4,7.0)
(5,5.25)
(6,4.5)
(1,3.4000000000000004)
     */

    //7.2使用indexAndSum和fiveDS得出聚类中心索引编号和5级标签tagsId的对应关系,即(predictIndex,tagsId)
    //将indexAndSum(普通集合)和fiveDS(转为普通集合)zip拉链即可
    val fiveRuleArr: Array[(Long, String)] = fiveDS.as[(Long,String)].collect().sortBy(_._2)
    //IndexedSeq[((predictIndex, sum), (tagsId, rule))]
    val tuples: immutable.IndexedSeq[((Int, Double), (Long, String))] = indexAndSum.zip(fiveRuleArr)
    //取出(predictIndex,tagsId)
    val indexAndTagsIdMap: Map[Int, Long] = tuples.map(t => {
      (t._1._1, t._2._1)
    }).toMap
    println("========================")
    indexAndTagsIdMap.foreach(println)

    /*
predictIndex,tagsId
     3        38
     0        39
     2        40
     4        41
     5        42
     6        43
     1        44
     */

    //8.取出predictResultDF中的userId和predict,并将predict转为tagsId,最后得出userId,tagsId
    val predict2tagsId = udf((predict:Int)=>{
      indexAndTagsIdMap(predict)
    })
    val newDF: DataFrame = predictResultDF.select('userId,predict2tagsId('predict) as "tagsId")
    newDF.show(false)
    /*
 +---------+------+
|userId   |tagsId|
+---------+------+
|13822725 |40    |
|13823083 |39    |
|138230919|39    |
|13823681 |40    |
|4033473  |39    |
     */

    //newDF
    null
  }
}
