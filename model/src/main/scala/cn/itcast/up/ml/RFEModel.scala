package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.common.HDFSUtils
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.collection.{immutable, mutable}

/**
 * Author itcast
 * Date 2020/3/23 15:30
 * Desc 使用KMeans算法+RFE打分计算用户活跃度模型
 */
object RFEModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  //45	活跃度		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_logs##family=detail##selectFields=global_user_id,loc_url,log_time		4	36
  override def getTagId(): Long = 45

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    //hbaseDF.show(false)
    //hbaseDF.printSchema()
    //fiveDS.show(false)
    //fiveDS.printSchema()

    /*
    用户id        访问的url                                  访问时间
 +--------------+-------------------------------------------+-------------------+
|global_user_id|loc_url                                    |log_time           |
+--------------+-------------------------------------------+-------------------+
|424           |http://m.eshop.com/mobile/coupon/getCoupons|2019-08-13 03:03:55|
|619           |http://m.eshop.com/?source=mobile          |2019-07-29 15:07:41|
|898           |http://m.eshop.com/mobile/item/11941.html  |2019-08-14 09:23:44|
|642           |http://www.eshop.com/l/2729-2931.html      |2019-08-11 03:20:17|
|130           |http://www.eshop.com/                      |2019-08-12 11:59:28|
|515           |http://www.eshop.com/l/2723-0-0-1-0-0-0-0-0|2019-07-23 14:39:25|
|274           |http://www.eshop.com/                      |2019-07-24 15:37:12|
|772           |http://ck.eshop.com/login.html             |2019-07-24 07:56:49|
|189           |http://m.eshop.com/mobile/item/9673.html   |2019-07-26 19:17:00|
|529           |http://m.eshop.com/mobile/search/_bplvbiwq_|2019-07-25 23:18:37|
|177           |http://m.eshop.com/mobile/cart/myCart.html |2019-07-23 21:01:26|
|247           |http://m.vip.eshop.com/mobile/member/toMemb|2019-07-22 06:58:05|
|702           |http://m.eshop.com/                        |2019-08-04 11:43:11|
|871           |http://vip.eshop.com/?ebi=ref-ixv5-1-hdv-3-|2019-08-11 09:38:00|
|349           |http://www.eshop.com/l/2725-2728.html?ebi=r|2019-07-22 12:17:54|
|538           |http://www.eshop.com/?cps_log_id=2015101200|2019-07-30 11:16:57|
|81            |http://www.eshop.com/product/11013.html?ebi|2019-08-06 09:10:37|
|308           |http://www.eshop.com/l/2811-2875-2877.html |2019-08-06 03:54:04|
|344           |http://m.eshop.com/?source=mobile          |2019-08-09 07:25:10|
|796           |http://member.eshop.com/login.html         |2019-08-07 15:16:28|
+--------------+-------------------------------------------+-------------------+
only showing top 20 rows

root
 |-- global_user_id: string (nullable = true)
 |-- loc_url: string (nullable = true)
 |-- log_time: string (nullable = true)

+---+----+
|id |rule|
+---+----+
|46 |1   |非常活跃
|47 |2   |活跃
|48 |3   |不活跃
|49 |4   |非常不活跃
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
     */

    //0.导入隐式转换,方便后续API调用
    import spark.implicits._
    import scala.collection.JavaConversions._
    import org.apache.spark.sql.functions._

    //0.定义常量字符串,避免后续拼写错误
    val recencyStr = "recency"
    val frequencyStr = "frequency"
    val engagementsStr = "engagements"
    val featureStr = "feature"
    val scaleFeatureStr = "scaleFeature"
    val predictStr = "predict"

    //1.计算用户的RFE
    //1.1声明RFE列对象
    //R:recency,最近/最大的访问时间距离今天的天数
    //F:frequency,最近一段时间访问的次数(直接求该用户全部访问次数)
    //E:engagements,最近一段时间页面互动度(直接求该用户全部访问页面数),注意实际中页面互动度,不同行业计算方式不一样,
    // 如电商:页面浏览时间、浏览商品数量;视频app:视频播放量,点赞量,转发量...
    val recencyColumn:Column = datediff(date_sub(current_date(),206),max('log_time)) as recencyStr
    val frequencyColumn:Column = count('loc_url) as frequencyStr//访问总次数
    val engagementsColumn:Column = countDistinct('loc_url) as engagementsStr//访问的总页面数,要去重

    //1.2分组计算各个用户的RFE的值
    val RFEDF: DataFrame = hbaseDF.groupBy('global_user_id as "userId")
      .agg(recencyColumn, frequencyColumn, engagementsColumn)
    //RFEDF.show(false)
    //RFEDF.printSchema()
    /*
 +------+-------+---------+-----------+
|userId|recency|frequency|engagements|
+------+-------+---------+-----------+
|296   |13     |380      |227        |
|467   |13     |405      |267        |
|675   |13     |370      |240        |
|691   |13     |387      |244        |

root
 |-- userId: string (nullable = true)
 |-- recency: integer (nullable = true)
 |-- frequency: long (nullable = false)
 |-- engagements: long (nullable = false)

     */

    //2.数据归一化/标准化
    // R:0-15天=5分，16-30天=4分，31-45天=3分，46-60天=2分，大于61天=1分
    // F:≥400=5分，300-399=4分，200-299=3分，100-199=2分，≤99=1分
    // E:≥250=5分，230-249=4分，210-229=3分，200-209=2分，1=1分
    val recencyScore: Column = when(col(recencyStr).between(0, 15), 5)
      .when(col(recencyStr).between(16, 30), 4)
      .when(col(recencyStr).between(31, 45), 3)
      .when(col(recencyStr).between(46, 60), 2)
      .when(col(recencyStr).gt(60), 1)
      .as(recencyStr)

    val frequencyScore: Column = when(col(frequencyStr).geq(400), 5)
      .when(col(frequencyStr).between(300, 399), 4)
      .when(col(frequencyStr).between(200, 299), 3)
      .when(col(frequencyStr).between(100, 199), 2)
      .when(col(frequencyStr).leq(99), 1)
      .as(frequencyStr)

    val engagementsScore: Column = when(col(engagementsStr).geq(250), 5)
      .when(col(engagementsStr).between(200, 249), 4)
      .when(col(engagementsStr).between(150, 199), 3)
      .when(col(engagementsStr).between(50, 149), 2)
      .when(col(engagementsStr).leq(49), 1)
      .as(engagementsStr)
    val RFEScoreDF: Dataset[Row] = RFEDF.select('userId, recencyScore, frequencyScore, engagementsScore)
      //对于日志表过滤掉可能为空的值,因为SparkMLlib的API不支持null
      .where('userId.isNotNull && 'recency.isNotNull && 'frequency.isNotNull && 'engagements.isNotNull)
    //RFEScoreDF.show(false)
    //RFEScoreDF.printSchema()

    //3.特征向量化
    val vectorDF: DataFrame = new VectorAssembler()
      .setInputCols(Array(recencyStr, frequencyStr, engagementsStr))
      .setOutputCol(featureStr)
      .transform(RFEScoreDF)
    //vectorDF.show(false)
    //vectorDF.printSchema()

    //4.选取K值
    /*println("k值选取开始")
    val ks:List[Int] = List(2,3,4,5,6,7,8,9)
    val map:mutable.Map[Int,Double] = mutable.Map[Int,Double]()
    for(k<-ks){
      val model: KMeansModel = new KMeans()
        .setK(k)
        .setSeed(100)
        .setMaxIter(20)
        .setFeaturesCol(featureStr)
        .setPredictionCol(predictStr)
        .fit(vectorDF)
      val sse: Double = model.computeCost(vectorDF)
      map.put(k,sse)
    }
    map.toArray.sortBy(_._1).foreach(println)
    println("k值选取结束")*/
    //假设上面的执行了,选取出的k为:4

    //5.模型的加载或训练再保存
    val  path: String = "/model/RFE35"
    var  model:KMeansModel = null
    if (HDFSUtils.getInstance().exists(path)){
      println("模型存在,直接加载并使用")
      model = KMeansModel.load(path)
    }else{
      println("模型不存在,重新训练并保存")
      model = new KMeans()
        .setK(4)
        .setSeed(100)
        .setMaxIter(20)
        .setFeaturesCol(featureStr)
        .setPredictionCol(predictStr)
        .fit(vectorDF)
      model.save(path)
    }

    //6.预测
    val predictResultDF: DataFrame = model.transform(vectorDF)
    predictResultDF.show(false)
   /* val RFMClusterInfo = predictResultDF
      .groupBy(predictStr)
      .agg(
        max(col(recencyStr) + col(frequencyStr) + col(monetaryStr)),
        min(col(recencyStr) + col(frequencyStr) + col(monetaryStr))
      )
      .sort(col(predictStr))//默认升序
    RFMClusterInfo.show(false)*/

    //7.将聚类中心编号和5级标签id对应起来
    //7.1将聚类中心编号和对应的RFE的和求出并按照RFE的和排序==>排好序的(index,RFESum)
    /*
    val centers: Array[linalg.Vector] = model.clusterCenters//获取所有聚类中心
    val indices: Range = centers.indices//获取聚类中心的索引编号组成的集合
    //IndexedSeq[(predict聚类中心索引编号,聚类中心的RFM的sum)]
    val indexAndSum: immutable.IndexedSeq[(Int, Double)] = indices.map(index => {
      val center: linalg.Vector = centers(index) //根据索引取聚类中心
      val RFMSum: Double = center.toArray.sum
      (index, RFMSum)
    }).sortBy(_._2).reverse//逆序
    indexAndSum.foreach(println)
     */
    /*
    //yield后面表示根据前面的循环变量生成一个集合
    val seq: immutable.IndexedSeq[(Int, Double)] = for(index<-model.clusterCenters.indices) yield (index,model.clusterCenters(index).toArray.sum)
    val indexAndSum: immutable.IndexedSeq[(Int, Double)] = seq.sortBy(_._2).reverse
     */
    val indexAndSum: immutable.IndexedSeq[(Int, Double)] = model.clusterCenters.indices.map(index => {
      (index, model.clusterCenters(index).toArray.sum)
    }).sortBy(_._2).reverse
    //indexAndSum.foreach(println)

    //7.2将indexAndSum(集合)和fiveDS(转为集合)进行zip拉链
    val fiveRuleArr: Array[(Long, String)] = fiveDS.as[(Long,String)].collect().sortBy(_._2)
    //tuples: immutable.IndexedSeq[((index, sum), (tagsId, rule))]
    val tuples: immutable.IndexedSeq[((Int, Double), (Long, String))] = indexAndSum.zip(fiveRuleArr)
    //Map[index就是predict编号, tagsId]
    val indexAndTagsIdMap: Map[Int, Long] = tuples.map(t => {
      (t._1._1, t._2._1)
    }).toMap

    //8.得出userId,tagsId
    //从predictResultDF中查询userId和predict,并将predcit转为tagsId
    val predict2TagsId =udf((predict:Int)=>{
      indexAndTagsIdMap(predict)
    })
    val newDF: DataFrame = predictResultDF.select('userId, predict2TagsId('predict) as "tagsId")
    newDF.show(false)
    /*
 +------+------+
|userId|tagsId|
+------+------+
|296   |49    |
|467   |46    |
|675   |49    |
|691   |49    |
|829   |46    |
|125   |49    |
     */

    null
  }
}
