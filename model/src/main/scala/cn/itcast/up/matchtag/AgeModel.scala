package cn.itcast.up.matchtag

import java.util.Properties

import cn.itcast.up.bean.HBaseMeta
import org.apache.commons.lang3.StringUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession, functions}

/**
 * Author itcast
 * Date 2020/3/18 9:40
 * Desc 完成年龄段标签/模型的开发
 */
object AgeModel {
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

    //2.读取和年龄段标签相关的4级标签rule并解析
    val fourRuleDS: Dataset[Row] = mysqlDF.select("rule").where("id=14")
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
    //fourRuleMap.foreach(println)

    //3.根据解析出来的ruel读取HBase数据
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
 +---+----------+
|id |birthday  |
+---+----------+
|1  |1992-05-31|
|10 |1980-10-13|
|100|1993-10-28|
|101|1996-08-18|
|102|1996-07-28|
|103|1987-05-13|

root
 |-- id: string (nullable = true)
 |-- birthday: string (nullable = true)
     */

    //4.读取和年龄段标签相关的5级标签(根据4级标签的id作为pid查询)
    val fiveDS: Dataset[Row] = mysqlDF.select("id","rule").where("pid=14")
    fiveDS.show(false)
    fiveDS.printSchema()
/*
+---+-----------------+
|id |rule             |
+---+-----------------+
|15 |19500101-19591231|
|16 |19600101-19691231|
|17 |19700101-19791231|
|18 |19800101-19891231|
|19 |19900101-19991231|
|20 |20000101-20091231|
|21 |20100101-20191231|
|22 |20200101-20291231|
+---+-----------------+
root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
 */
    //5.根据HBase数据和5级标签数据进行匹配,得出userId,tagsId
    //5.1统一格式,将1999-09-09统一为:19990909
    import org.apache.spark.sql.functions._
    //regexp_replace(hbaseDF.col("birthday"),"-","")
    //声明birthdayColumn这一列怎么来,并没有真正执行,怎么执行?需要进行查询
    //val birthdayColumn: Column = regexp_replace('birthday,"-","")
    val hbaseDF2: DataFrame = hbaseDF.select('id as "userId",regexp_replace('birthday,"-","") as "birthday")
    hbaseDF2.show(false)
    /*
 +------+--------+
|userId|birthday|
+------+--------+
|1     |19920531|
|10    |19801013|
|100   |19931028|
|101   |19960818|
+------+--------+
     */

    //5.2将fiveDS拆分为("tagsId","start","end")
    //fiveDS.as[(tagsId,rule)]
    val fiveDS2: DataFrame = fiveDS.as[(Long, String)].map(t => {
      val arr: Array[String] = t._2.split("-")
      (t._1, arr(0), arr(1))
    }).toDF("tagsId", "start", "end")
    fiveDS2.show(false)
    /*
 +------+--------+--------+
|tagsId|start   |end     |
+------+--------+--------+
|17    |19700101|19791231|
|18    |19800101|19891231|
|19    |19900101|19991231|
     */

    //5.3将hbaseDF2和fiveDS2直接join
    val newDF: DataFrame = hbaseDF2.join(fiveDS2) //join默认为inner
      .where(hbaseDF2.col("birthday").between(fiveDS2.col("start"), fiveDS2.col("end")))
      .select(hbaseDF2.col("userId"), fiveDS2.col("tagsId"))
    newDF.show(false)



    //6.查询HBase中的oldDF
    val oldDF: DataFrame = spark.read
      .format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.INTYPE, hbaseMeta.inType)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.HBASETABLE, "test")
      .option(HBaseMeta.SELECTFIELDS, "userId,tagsId")
      .load()

    //7.合并newDF和oldDF
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

    //8.将最终结果写到HBase
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
