package cn.itcast.up.matchtag

import java.util.Properties

import cn.itcast.up.bean.HBaseMeta
import org.apache.commons.lang3.StringUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * Author itcast
 * Date 2020/3/16 15:36
 * Desc 职业标签/模型/任务开发
 */
object JobModel {
  def main(args: Array[String]): Unit = {
    //0.准备Spark开发环境
    val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("model")
      .getOrCreate()
    val sc: SparkContext = spark.sparkContext
    sc.setLogLevel("WARN")
    //导入隐式转换
    import spark.implicits._

    //1.读取MySQL中的数据
    val url:String = "jdbc:mysql://bd001:3306/tags_new?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC&user=root&password=123456"
    val tableName:String = "tbl_basic_tag"
    val properties:Properties = new Properties()
    val mysqlDF: DataFrame = spark.read.jdbc(url,tableName,properties)
    //mysqlDF.show(false)

    //2.读取和职业标签相关的4级标签rule并解析
    val fourRuleDS: Dataset[Row] = mysqlDF.select("rule").where("id=7")
    //inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_users##family=detail##selectFields=id,job
    //解析rule为map
    val fourRuleMap: Map[String, String] = fourRuleDS.map(row => {
      val rowStr: String = row.getAs[String]("rule")
      val kvs: Array[String] = rowStr.split("##")
      kvs.map(kvStr => {
        val kv: Array[String] = kvStr.split("=")
        (kv(0), kv(1))
      })
    }).collectAsList().get(0).toMap

    //3.根据4级标签加载HBase数据
    val hbaseMeta = HBaseMeta(fourRuleMap)

    val hbaseDF: DataFrame = spark.read
      .format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.INTYPE, hbaseMeta.inType)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.HBASETABLE, hbaseMeta.hbaseTable)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.SELECTFIELDS, hbaseMeta.selectFields)
      .load()
    hbaseDF.show(false)
    hbaseDF.printSchema()
    /*
 +---+---+
|id |job|
+---+---+
|1  |3  |
|10 |5  |
|100|3  |
|101|1  |
|102|1  |
|103|3  |
|104|6  |
|105|2  |
|106|4  |
|107|1  |
|108|4  |
|109|6  |
|11 |6  |
|110|4  |
|111|1  |
|112|1  |
|113|6  |
|114|1  |
|115|4  |
|116|6  |
+---+---+
only showing top 20 rows

root
 |-- id: string (nullable = true)
 |-- job: string (nullable = true)
     */

    //4.读取和职业标签相关的5级标签(根据4级标签的id作为pid查询)
    val fiveDS: Dataset[Row] = mysqlDF.select("id","rule").where("pid=7")
    fiveDS.show(false)
    fiveDS.printSchema()
    /*
 +---+----+
|id |rule|
+---+----+
|8  |1   |
|9  |2   |
|10 |3   |
|11 |4   |
|12 |5   |
|13 |6   |
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
     */

    //5.根据HBase数据和5级标签数据进行匹配,得出userId,tagsId
    //fiveRuleMap: Map[rule(job), tagId]
    val fiveRuleMap: Map[String, Long] = fiveDS.as[(Long, String)].map(t => {
      (t._2, t._1)
    }).collect().toMap

    //编写sql使用UDF将job转为tagId
    import org.apache.spark.sql.functions._
    val job2tagId: UserDefinedFunction = udf(
        (job:String)=>{
          val tagId: Long = fiveRuleMap(job)
          tagId
      }
    )

    val newDF: DataFrame = hbaseDF.select('id as "userId",job2tagId('job) as "tagsId")
    newDF.show(false)

    /*
newDF用户职业
+------+------+
|userId|tagsId|
+------+------+
|1     |10    |
|10    |12    |
|100   |10    |
|101   |8     |
|102   |8     |
|103   |10    |
|104   |13    |
|105   |9     |
|106   |11    |
|107   |8     |
|108   |11    |
|109   |13    |
|11    |13    |
|110   |11    |
|111   |8     |
|112   |8     |
|113   |13    |
|114   |8     |
|115   |11    |
|116   |13    |
+------+------+
     */


    //6.将userId,tagsId保存到HBase
    //注意:这里不能简单的直接将结果写入到HBase,因为如果直接写入的话会导致将HBase中的之前的标签如性别标签覆盖了
    //而我们想要的是既保留用户的性别标签也保留用户的职业标签
    //思考该如何去做?
    //方案1:不同类型的标签存储到不同的列族,同一个列族下不同的标签存储在不同的列,到达应该存储到哪一个列族/哪一个列,根据业务和场景进行划分
    //如:性别/年龄/职业放到人口属性列族下的三个不同的列中,消费周期/支付方式放到商业属性下的两个不同的列
    //(但是要求提前有划分标准,后续保存的时候严格的按照标准来)
    //方案2:直接将HBase画像结果表的历史数据和当前这一次的结果数据合并，最后将合并的结果覆盖存储
    //该方案简单一点,所以学习时直接使用它
    //(但是注意,该方案,当数量量很大,标签很多的时候,不方便管理,不方便后续的查询)

    //6.1查询HBase中原来的画像结果表
    val oldDF: DataFrame = spark.read
      .format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.INTYPE, hbaseMeta.inType)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.HBASETABLE, "test")
      .option(HBaseMeta.SELECTFIELDS, "userId,tagsId")
      .load()
    oldDF.show(false)
    /*
 oldDF用户性别
+------+------+
|userId|tagsId|
+------+------+
|1     |6     |
|10    |6     |
|100   |6     |
|101   |5     |
|102   |6     |
|103   |5     |
|104   |5     |
|105   |6     |
|106   |5     |
|115   |5     |
|116   |6     |
+------+------+
     */

    //6.2将oldDF和newDF进行合并--左外连接
    /*
+------+------+
|userId|tagsId|
+------+------+
|1     |6,10  |
|10    |6,12  |
|100   |6,10  |
|101   |5,8   |
|102   |6,8   |
|103   |5,10  |
     */

    //SQL方式newDF左外连接oldDF
    newDF.createOrReplaceTempView("t_new")
    oldDF.createOrReplaceTempView("t_old")

    //自定义sql风格udf
    /*spark.udf.register("merge",(newTagsId:String,oldTagsId:String)=>{
        if(StringUtils.isBlank(newTagsId)){
          oldTagsId
        }else if(StringUtils.isBlank(oldTagsId)){
          newTagsId
        }else{
          //set可以自动去重
          //(newTagsId.split(",") ++ oldTagsId.split(",")).toSet.mkString(",")
          val newArr: Array[String] = newTagsId.split(",")
          val oldArr: Array[String] = oldTagsId.split(",")
          val resultArr: Array[String] = newArr ++ oldArr
          val set: Set[String] = resultArr.toSet
          val result: String = set.mkString(",")
          result
        }
    })

    val sql =
      """
        |select n.userId,merge(n.tagsId,o.tagsId) as tagsId
        |from t_new n
        |left join t_old o
        |on n.userId = o.userId
        |""".stripMargin

    val resultDF: DataFrame = spark.sql(sql)
    resultDF.show(false)*/
    /*
 +------+------+
|userId|tagsId|
+------+------+
|296   |13,5  |
|467   |13,6  |
|675   |10,6  |
|691   |8,5   |
|829   |12,5  |
     */



    //DSL方式newDF左外连接oldDF---课后思考下
    //自定义DSL风格的UDF
    val meger = udf((newTagsId:String,oldTagsId:String)=>{
      if(StringUtils.isBlank(newTagsId)){
        oldTagsId
      }else if(StringUtils.isBlank(oldTagsId)){
        newTagsId
      }else{
        //set可以自动去重
        //(newTagsId.split(",") ++ oldTagsId.split(",")).toSet.mkString(",")
        val newArr: Array[String] = newTagsId.split(",")
        val oldArr: Array[String] = oldTagsId.split(",")
        val resultArr: Array[String] = newArr ++ oldArr
        val set: Set[String] = resultArr.toSet
        val result: String = set.mkString(",")
        result
      }
    })

    val resultDF: DataFrame = newDF.join(
      oldDF, //和谁join
      newDF.col("userId") === oldDF.col("userId"), //join条件
      "left" //join的类型,默认是inner
    ).select(
      newDF.col("userId"),
      meger(newDF.col("tagsId"), oldDF.col("tagsId")) as "tagsId"
    )
    resultDF.show(false)

    resultDF.write
        .format("cn.itcast.up.tools.HBaseSource")
        .option(HBaseMeta.ZKHOSTS,hbaseMeta.zkHosts)
        .option(HBaseMeta.ZKPORT,hbaseMeta.zkPort)
        .option(HBaseMeta.HBASETABLE,"test")
        .option(HBaseMeta.FAMILY,hbaseMeta.family)
        .option(HBaseMeta.SELECTFIELDS,"userId,tagIds")
        .option(HBaseMeta.ROWKEY,"userId")
        .save()

  }
}
