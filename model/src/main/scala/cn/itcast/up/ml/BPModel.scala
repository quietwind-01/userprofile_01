package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.ml.RFMModel3.spark
import com.mysql.cj.protocol.a.NativeConstants.IntegerDataType
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * Author itcast
 * Date 2020/3/28 14:41
 * Desc 使用基于隐语义模型的协同过滤-ALS推荐算法完成用户购物偏好/品牌偏好/类别偏好/味道偏好/菜系偏好模型开发
 * BP--BrandPreference
 */
object BPModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  //60	购物偏好模型		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_logs##family=detail##selectFields=global_user_id,loc_url,log_time		4	36
  override def getTagId(): Long = 60

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    hbaseDF.show(false)
    hbaseDF.printSchema()
    //fiveDS不用,因为商品太多,标签表中不用存储,最后结果直接给用户打商品id/品牌id即可
/*
用户id          访问的url                                  访问时间
+--------------+-----------------------------------------+-------------------+
|global_user_id|loc_url                                  |log_time           |
+--------------+-----------------------------------------+-------------------+
|538           |http://www.eshop.com/?cps_log_id=20151012|2019-07-30 11:16:57|
|81            |http://www.eshop.com/product/11013.html?e|2019-08-06 09:10:37|
|308           |http://www.eshop.com/l/2811-2875-2877.htm|2019-08-06 03:54:04|
|344           |http://m.eshop.com/?source=mobile        |2019-08-09 07:25:10|
|796           |http://member.eshop.com/login.html       |2019-08-07 15:16:28|
+--------------+-----------------------------------------+-------------------+
only showing top 20 rows

root
 |-- global_user_id: string (nullable = true)
 |-- loc_url: string (nullable = true)
 |-- log_time: string (nullable = true)
 */
    //0.导入隐式转换
    import org.apache.spark.sql.functions._
    import spark.implicits._

    //注意:通过观察上面的数据,只有用户对商品的url的访问信息,没有显式的评分数据,所以我们这里只能用隐式评分
    //也就是将用户对商品的点击/浏览等操作当作是隐式的评分数据

    //1.解析数据中的商品id
    //如:从数据:http://www.eshop.com/product/11013.html?中解析出商品id:11013
    val url2productId = udf((url:String)=>{
      var productId:String = null
      if (url.contains("/product/") && url.contains(".html")){
        val start: Int = url.indexOf("/product/")
        val end: Int = url.indexOf(".html")
        if(end>start){
          productId = url.substring(start+9,end)
        }
      }
      productId
    })

    val tempDF: Dataset[Row] = hbaseDF.select('global_user_id as "userId", url2productId('loc_url) as "productId")
      .filter('productId.isNotNull && 'userId.isNotNull)
    tempDF.show(false)
    tempDF.printSchema()
    //userId,productId
    /*
 +------+---------+
|userId|productId|
+------+---------+
|81    |11013    |
|767   |11813    |
|302   |5353     |
|370   |9221     |
|405   |4167     |
     */

    val tempDF2: DataFrame = tempDF.groupBy('userId, 'productId)
      .agg(count('productId).as("rating"))
    //2.得出用户id、商品id、评分(用隐式评分,访问一次就算1分)
    val ratingDF: Dataset[Row] = tempDF2.select('userId.cast(IntegerType), 'productId.cast(IntegerType),'rating.cast(DoubleType))
      .filter('productId.isNotNull && 'userId.isNotNull && 'rating.isNotNull)//防止转换失败有null
    ratingDF.show(false)
    ratingDF.printSchema()
    //用户id,商品id,隐式评分
    //userId,productId,rating
    /*
 +------+---------+------+
|userId|productId|rating|
+------+---------+------+
|533   |11455    |1.0   |
|322   |11949    |1.0   |
|255   |10243    |1.0   |
|266   |6038     |2.0   |
|626   |9371     |3.0   |
+------+---------+------+
only showing top 20 rows

root
 |-- userId: integer (nullable = true)
 |-- productId: integer (nullable = true)
 |-- rating: double (nullable = false)
     */

    //3.构建ALS进行训练--可以理解为矩阵分解
    val model: ALSModel = new ALS()
      .setUserCol("userId") //设置用户id是哪一列
      .setItemCol("productId") //设置商品id是哪一列
      .setRatingCol("rating") //设置评分是哪一列
      .setImplicitPrefs(true) //设置是否使用隐式评分,默认是false,这里应该为true,因为我们使用的是隐式评分
      .setRank(10) //设置K值,也即是隐藏的特征的数量
      .setMaxIter(10) //设置最大迭代次数
      .setAlpha(0.1) //设置迭代步长
      .fit(ratingDF)

    //4.预测
    //预测所有用户可能感兴趣的5个商品(对所有用户推荐)
    val result: DataFrame = model.recommendForAllUsers(5)
    result.show(false)
    result.printSchema()
    //userId,[给该用户推荐的5个商品的集合]
    /*
 +------+-------------------------------------------------------------------------------------------------+
|userId|recommendations                                                                                  |
+------+-------------------------------------------------------------------------------------------------+
|471   |[[10935,0.62081915], [6603,0.59677047], [9371,0.55584806], [6393,0.49544087], [6395,0.4918112]]  |
|463   |[[10935,0.5040521], [6603,0.49695188], [9371,0.4754672], [7173,0.44227752], [6393,0.4335229]]    |
|833   |[[6603,0.5791473], [10935,0.5648133], [9371,0.5542244], [7173,0.51665217], [6393,0.5120446]]     |
|496   |[[6603,0.6095567], [10935,0.552063], [9371,0.5505363], [6393,0.5156756], [6395,0.49742582]]      |

root
 |-- userId: integer (nullable = false)
 |-- recommendations: array (nullable = true)
 |    |-- element: struct (containsNull = true)
 |    |    |-- productId: integer (nullable = true)
 |    |    |-- rating: float (nullable = true)
     */


    //5.得出newDF:userId,tagIds
    //也即是要解析上面的数据,将userId,recommendations变为:userId,tagIds
    //471   [[10935,0.62081915], [6603,0.59677047], [9371,0.55584806], [6393,0.49544087], [6395,0.4918112]]
    //变为:
    //471   10935,6603,9371,6393,6395
    //.as[(用户id, Array[(商品id, 评分)])]
    val newDF: DataFrame = result.as[(Int, Array[(Int, Double)])].map(t => {
      val userId: Int = t._1
      val arr: Array[(Int, Double)] = t._2
      val productIds: Array[Int] = arr.map(_._1)
      val tagIds: String = productIds.mkString(",")
      (userId, tagIds)
    }).toDF("userId", "tagIds")
    newDF.show(false)
    /*
+------+---------------------------+
|userId|tagIds                     |
+------+---------------------------+
|471   |10935,6603,9371,6393,6395  |
|463   |10935,6603,9371,7173,6393  |
|833   |6603,10935,9371,7173,6393  |
|496   |6603,10935,9371,6393,6395  |
|148   |6603,10935,9371,7173,6393  |
|540   |6603,10935,9371,6393,6395  |
|392   |10935,9371,6603,7173,10781 |
     */

    null
  }
}
