package cn.itcast.up.matchtag

import java.util
import java.util.Properties

import cn.itcast.up.bean.HBaseMeta
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * Author itcast
 * Date 2020/3/16 10:32
 * Desc 性别标签/模型/任务开发
 */
object GenderModel {
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

    //1.读取MySQL数据
    val url:String = "jdbc:mysql://bd001:3306/tags_new?useUnicode=true&characterEncoding=UTF-8&serverTimezone=UTC&user=root&password=123456"
    val tableName:String = "tbl_basic_tag"
    val properties:Properties = new Properties()
    val mysqlDF: DataFrame = spark.read.jdbc(url,tableName,properties)
    //mysqlDF.show(false)

    //2.读取需要的性别标签相关的4级标签的rule并解析rule
    //2.1读取4级标签的rule
    val fourRuleDS: Dataset[Row] = mysqlDF.select("rule").where("id=4")
    //val fourRuleDS: Dataset[Row] = mysqlDF.select("rule").where($"id"===4)
    //val fourRuleDS: Dataset[Row] = mysqlDF.select("rule").where('id===4)
    //fourRuleDS.show(false)
    //+-------------------------------------------------------------------------------------------------------------+
    //|rule                                                                                                         |
    //+-------------------------------------------------------------------------------------------------------------+
    //|inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_users##family=detail##selectFields=id,gender|
    //+-------------------------------------------------------------------------------------------------------------+

    //2.2解析rule为map
    val kvtupleDS: Dataset[Array[(String, String)]] = fourRuleDS.map(row => {
      //将Row转为String
      val ruleStr: String = row.getAs[String]("rule")
      val kvs: Array[String] = ruleStr.split("##")
      val kvtuple: Array[(String, String)] = kvs.map(kvStr => {
        val kv: Array[String] = kvStr.split("=")
        (kv(0), kv(1))
      })
      kvtuple
    })
    val kvtupleList: util.List[Array[(String, String)]] = kvtupleDS.collectAsList()
    val kvtupleArr: Array[(String, String)] = kvtupleList.get(0)
    val fourRuleMap: Map[String, String] = kvtupleArr.toMap
    //fourRuleMap.foreach(println)
    /*
(selectFields,  id,gender)
(inType,HBase)
(zkHosts,192.168.10.20)
(zkPort,2181)
(hbaseTable,tbl_users)
(family,detail)
     */

    //3.根据解析好的4级标签的rule加载HBase的数据
    //在这里可以使用Spark提供的原始的操作HBase的API(很难用),也可以使用封装好的自定义的HBase数据源(也可以使用开源的)
    //3.1准备HBase连接参数(将上面的map转为HBaseMeta方便后面操作,本质上就是将map封装为样例类对象)
    val hbaseMeta: HBaseMeta = HBaseMeta(fourRuleMap)
    //3.2使用自定义HBase数据源根据hbaseMeta加载HBase数据
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
 +---+------+
|id |gender|
+---+------+
|1  |2     |
|10 |2     |
|100|2     |
|101|1     |
|102|2     |
|103|1     |
|104|1     |
|105|2     |
|106|1     |
|107|1     |
|108|1     |
|109|1     |
|11 |2     |
|110|2     |
|111|1     |
|112|2     |
|113|1     |
|114|1     |
|115|1     |
|116|2     |
+---+------+
only showing top 20 rows

root
 |-- id: string (nullable = true)
 |-- gender: string (nullable = true)
     */

    //4.读取MySQL中和性别标签相关的5级标签(根据4级标签的id作为pid查询)
    val fiveDS: Dataset[Row] = mysqlDF.select("id","rule").where('pid===4)
    fiveDS.show(false)
    fiveDS.printSchema()
    /*
 +---+----+
|id |rule|
+---+----+
|5  |1   |
|6  |2   |
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
     */

    //5.将hbaseDF和fiveDS进行匹配,得到如下的数据:
    /*
+---+------+
|id |gender|tagid
+---+------+
|1  |2     |6
|101|1     |5
     */
     //5.1将fiveDF转为map,方便后续自定义UDF操作
    //fiveDS[tagid,rule]转为:Map[rule,tagid]
    val tempDS: Dataset[(String, Long)] = fiveDS.as[(Long, String)].map(t => {
      (t._2, t._1)
    })
    val tempArr: Array[(String, Long)] = tempDS.collect()
    //Map[rule, tagid]
    val fiveMap: Map[String, Long] = tempArr.toMap

    //5.2使用单表+UDF完成hbaseDF和fiveDS的匹配
    //自定义DSL风格的udf,将gender转为tagid
    import org.apache.spark.sql.functions._
    val gender2tagid = udf((gender:String)=>{
      fiveMap(gender)
    })

    val resultDF: DataFrame = hbaseDF.select('id as "userId",gender2tagid('gender) as "tagsId")
    resultDF.show(false)
    /*
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

    //6.最终将如下数据保存到HBase的画像结果表中:
    /*
    userId,tagsId
    1       6
    101     5
     */
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
