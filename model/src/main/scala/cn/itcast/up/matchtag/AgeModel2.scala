package cn.itcast.up.matchtag

import cn.itcast.up.base.BaseModel
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/**
 * Author itcast
 * Date 2020/3/18 9:40
 * Desc 完成年龄段标签/模型的开发
 */
object AgeModel2 extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()//调用从父类继承的execute方法
  }

  override def getTagId(): Long = 14

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    println("执行子类的compute方法")
    //5.1统一格式,将1999-09-09统一为:19990909
    import spark.implicits._
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
|18    |19800101|19891231|
|19    |19900101|19991231|
     */

    //5.3将hbaseDF2和fiveDS2直接join
    val newDF: DataFrame = hbaseDF2.join(fiveDS2) //join默认为inner
      .where(hbaseDF2.col("birthday").between(fiveDS2.col("start"), fiveDS2.col("end")))
      .select(hbaseDF2.col("userId"), fiveDS2.col("tagsId"))
    newDF.show(false)

    newDF
  }
}
