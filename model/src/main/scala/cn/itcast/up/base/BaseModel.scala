package cn.itcast.up.base

import java.util.Properties

import cn.itcast.up.bean.HBaseMeta
import org.apache.commons.lang3.StringUtils
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * Author itcast
 * Date 2020/3/18 10:57
 * Desc trait相当于java中的接口,但是支持普通方法(可以被子类/实现类继承),还支持抽象方法(可以让子类重写/实现)
 * 我们这里使用trait,就可以将重复的代码直接封装在trait的普通方法中,将不一样的步骤/代码做出抽象方法让子类/实现类去重写/实现
 * 模版方法设计模式
 */
trait BaseModel {

  //声明一个抽象方法,由子类提供方法实现
  def getTagId():Long
  //抽象方法,由子类提供具体的方法实现
  def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame

  //现在execute方法中的代码太多了,而且第5步应该由子类来实现,所以execute方法中不应该直接一撸到底全写完,而应该将第5步留给子类实现
  //其他步骤全写在execute方法中也太多了,所以可以再封装为方法,或提前出去
  def execute():Unit = {
    println("execute方法执行了,依次执行1~8步,遇到抽象的执行子类的")
    //1.读取MySQL中的数据
    val mysqlDF: DataFrame = getMySQLData()

    //2.读取模型/标签相关的4级标签rule并解析--标签id不一样
    val id = getTagId()
    val hbaseMeta: HBaseMeta = getFourRule(mysqlDF,id)

    //3.根据解析出来的ruel读取HBase数据
    val hbaseDF: DataFrame = getHBaseDF(hbaseMeta)

    //4.读取模型/标签相关的5级标签(根据4级标签的id作为pid查询)---标签id不一样
    val fiveDS: Dataset[Row] = getFiveRuleDF(mysqlDF,id)

    //5.根据HBase数据和5级标签数据进行匹配,得出userId,tagsId---实现代码不一样
    val newDF: DataFrame = compute(hbaseDF,fiveDS)

    //6.查询HBase中的oldDF
    val oldDF: DataFrame = getHBaseOldDF(hbaseMeta)

    //7.合并newDF和oldDF
    val resultDF: DataFrame = merge(newDF,oldDF)

    //8.将最终结果写到HBase
    save2HBase(resultDF,hbaseMeta)
  }

  //0.准备Spark开发环境
  val spark: SparkSession = SparkSession.builder()
    .master("local[*]")
    .appName("model")
    .getOrCreate()
  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("WARN")
  //导入隐式转换
  import spark.implicits._


  //哪些是绝大部分代码重复的?-01234678--如何封装抽取?子类如何使用?--课下思考一下
  //首先,0~8这些步骤是一个标签/模型开发的通用步骤,必须得执行,那么如果让子类一个个的调用,那么得调用8次啊!
  //所以应该将这个8个步骤放到1个execute方法中,让子类直接调用这1个execute方法即可

  def getMySQLData(): DataFrame = {
    val url:String = "jdbc:mysql://bd001:3306/tags_new?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC&user=root&password=123456"
    val tableName:String = "tbl_basic_tag"
    val properties:Properties = new Properties()
    val mysqlDF: DataFrame = spark.read.jdbc(url,tableName,properties)
    mysqlDF
  }

  def getFourRule(mysqlDF: DataFrame,id:Long): HBaseMeta = {
    val fourRuleDS: Dataset[Row] = mysqlDF.select("rule").where('id === id)
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
    val hbaseMeta = HBaseMeta(fourRuleMap)
    hbaseMeta
  }

  def getHBaseDF(hbaseMeta: HBaseMeta): DataFrame = {
    spark.read.format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.INTYPE, hbaseMeta.inType)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.HBASETABLE, hbaseMeta.hbaseTable)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.SELECTFIELDS, hbaseMeta.selectFields)
      .load()
  }

  def getFiveRuleDF(mysqlDF: DataFrame, id: Long): Dataset[Row] = {
    mysqlDF.select("id","rule").where('pid===id)
  }

  def getHBaseOldDF(hbaseMeta: HBaseMeta): DataFrame = {
    spark.read
      .format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.INTYPE, hbaseMeta.inType)
      .option(HBaseMeta.ZKHOSTS, hbaseMeta.zkHosts)
      .option(HBaseMeta.ZKPORT, hbaseMeta.zkPort)
      .option(HBaseMeta.FAMILY, hbaseMeta.family)
      .option(HBaseMeta.HBASETABLE, "test")
      .option(HBaseMeta.SELECTFIELDS, "userId,tagsId")
      .load()
  }

  def merge(newDF: DataFrame, oldDF: DataFrame): DataFrame = {
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
    resultDF
  }

  def save2HBase(resultDF: DataFrame,hbaseMeta: HBaseMeta): Unit = {
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
