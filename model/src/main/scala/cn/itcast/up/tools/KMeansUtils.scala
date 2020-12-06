package cn.itcast.up.tools

import cn.itcast.up.common.HDFSUtils
import cn.itcast.up.ml.PSMModel.spark
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.{immutable, mutable}

/**
 * Author itcast
 * Date 2020/3/24 11:34
 * Desc 
 */
object KMeansUtils {
  def getNewDF(predictResultDF: DataFrame, indexAndTagsIdMap: Map[Int, Long]): DataFrame = {
    import spark.implicits._
    import scala.collection.JavaConversions._
    import org.apache.spark.sql.functions._
    //得出userId,tagsId
    //从predictResultDF中查询userId和predict,并将predcit转为tagsId
    val predict2TagsId =udf((predict:Int)=>{
      indexAndTagsIdMap(predict)
    })
    val newDF: DataFrame = predictResultDF.select('userId, predict2TagsId('predict) as "tagsId")
    newDF
  }
  def getIndexAndTagsIdMap(model: KMeansModel, fiveDS: Dataset[Row]): Map[Int, Long] = {
    import spark.implicits._
    import scala.collection.JavaConversions._
    import org.apache.spark.sql.functions._
    //将聚类中心编号和对应的RFE的和求出并按照RFE的和排序==>排好序的(index,RFESum)
    val indexAndSum: immutable.IndexedSeq[(Int, Double)] = model.clusterCenters.indices.map(index => {
      (index, model.clusterCenters(index).toArray.sum)
    }).sortBy(_._2).reverse
    //indexAndSum.foreach(println)

    //将indexAndSum(集合)和fiveDS(转为集合)进行zip拉链
    val fiveRuleArr: Array[(Long, String)] = fiveDS.as[(Long,String)].collect().sortBy(_._2)
    //tuples: immutable.IndexedSeq[((index, sum), (tagsId, rule))]
    val tuples: immutable.IndexedSeq[((Int, Double), (Long, String))] = indexAndSum.zip(fiveRuleArr)
    //Map[index就是predict编号, tagsId]
    val indexAndTagsIdMap: Map[Int, Long] = tuples.map(t => {
      (t._1._1, t._2._1)
    }).toMap
    indexAndTagsIdMap
  }

  def getKMeansModel(path:String,k:Int,maxIter:Int,vectorDF:DataFrame): KMeansModel = {
    var model: KMeansModel = null
    if (HDFSUtils.getInstance().exists(path)) {
      println("模型存在,直接加载并使用")
      model = KMeansModel.load(path)
    } else {
      println("模型不存在,重新训练并保存")
      model = new KMeans()
        .setK(k)
        .setSeed(100)
        .setMaxIter(maxIter)
        .setFeaturesCol("feature")
        .setPredictionCol("predict")
        .fit(vectorDF)
      model.save(path)
    }
    model
  }

  def selectK(vectorDF: DataFrame, ks: List[Int]): Int = {
    /*val map: mutable.Map[Int, Double] = mutable.Map[Int, Double]()
    for (k <- ks) {
      val model: KMeansModel = new KMeans()
        .setK(k)
        .setSeed(100)
        .setMaxIter(20)
        .setFeaturesCol("feature")
        .setPredictionCol("predict")
        .fit(vectorDF)
      val sse: Double = model.computeCost(vectorDF)
      map.put(k, sse)
    }
    map.toArray.sortBy(_._1).foreach(println)*/
    5
  }

  def getVecotrDF(PSMDF: Dataset[Row], inputCols: Array[String]): DataFrame = {
    //特征向量化
    val vectorDF: DataFrame = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("feature")
      .transform(PSMDF)
    vectorDF
  }
}
