package cn.itcast.up.statistics

import cn.itcast.up.base.BaseModel
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

/**
 * Author itcast
 * Date 2020/3/18 14:21
 * Desc 完成统计型标签-消费周期标签/模型的开发
 * 需求:获取用户最近一次消费时间距离今天的天数,
 * 如:是5天前,一个月前消费的
 * 方便找出长时间未消费的用户,做用户流失预警/用户挽回
 */
object CycleModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }

  //23	消费周期		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_orders##family=detail##selectFields=memberId,finishTime		4	3
  override def getTagId(): Long = 23

  //该方法由子类实现,完成用户消费周期标签/模型的计算
  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    //fiveDS.show(false)
    //fiveDS.printSchema()
    /*
 +---+-----+
|id |rule |
+---+-----+
|24 |0-7  |
|25 |8-14 |
|26 |15-30|
|27 |31-60|
|28 |61-90|
+---+-----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)

     */

    //hbaseDF.show(false)
    //hbaseDF.printSchema()
    /*
    +---------+----------+
    |memberId |finishTime|
    +---------+----------+
    |13823431 |1564415022|
    |13823431  |1565687310|
    |4035291  |1564681801|
    |4035041  |1565799378|
    |13823285 |1565062072|
    |4034219  |1563601306|
    |138230939|1565509622|
    |4035083  |1565731851|
    |138230935|1565382991|
    |13823231 |1565677650|
    |47       |1563827657|
    |13822713 |1565244075|
    |13823585 |1564108078|
    |4033345  |1564719081|
    |13823271 |1565429547|
    |13823325 |1563985505|
    |4033693  |1563850122|
    |13822859 |1563863380|
    |13823431 |1565749812|
    |13823587 |1564845843|
    +---------+----------+
    only showing top 20 rows

    root
     |-- memberId: string (nullable = true)
     |-- finishTime: string (nullable = true)
     */
    //0.导入隐式转换
    import spark.implicits._
    import org.apache.spark.sql.functions._

    //1.求用户最近一次消费时间
    //根据用户分组取finishTime最大值即可
    val tempDF: DataFrame = hbaseDF
      .groupBy('memberId as "userId")
      .agg(max('finishTime) as "maxFinishTime")
    //tempDF.show(false)

    /*
+---------+----------+
|userId   |maxFinishTime|
+---------+----------+
|13822725 |1566056954|
|13823083 |1566048648|
|138230919|1566012606|
|13823681 |1566012541|
|4033473  |1566022264|
     */

    //2.求用户最近一次消费时间距离今天的天数
    //from_unixtime:将maxFinishTime转为时间对象
    //current_date:获取当前时间对象
    //datediff:求当前时间和maxFinishTime时间的天数差
    //date_sub:表示把时间减去多少天
    //声明daysCoulmn该如何计算,但是并没有真正的执行,得去select才可以
    val daysCoulmn: Column = datediff(date_sub(current_date(),200),from_unixtime('maxFinishTime)) as "days"
    val tempDF2: DataFrame = tempDF.select('userId,daysCoulmn)
    tempDF2.show(false)
    /*
+---------+----+
|userId   |days|
+---------+----+
|13822725 |14 |
|13823083 |14 |
|138230919|14 |
|13823681 |14 |
|4033473  |14 |
|13822841 |14 |
|13823153 |14 |
|13823431 |14 |
|4033348  |14 |
     */

    //3.将fiveDS拆成:("tagsId","start","end")
    val fiveDS2: DataFrame = fiveDS.as[(Long, String)].map(t => {
      val arr: Array[String] = t._2.split("-")
      (t._1, arr(0), arr(1))
    }).toDF("tagsId", "start", "end")
    fiveDS2.show(false)
    /*
 +------+-----+---+
|tagsId|start|end|
+------+-----+---+
|24    |0    |7  |
|25    |8    |14 |
|26    |15   |30 |
|27    |31   |60 |
|28    |61   |90 |
+------+-----+---+
     */

    //4.将tempDF2和fiveDS2进行匹配
    val newDF: DataFrame = tempDF2.join(fiveDS2)
      .where(tempDF2.col("days").between('start, 'end))
      .select('userId, 'tagsId)
    newDF.show(false)
    /*
 +---------+------+
|userId   |tagsId|
+---------+------+
|13822725 |25    |
|13823083 |25    |
|138230919|25    |
|13823681 |25    |
|4033473  |25    |
|13822841 |25    |
|13823153 |25    |
|13823431 |25    |
|4033348  |25    |
|4033483  |25    |
|4033575  |25    |
|4034191  |25    |
|4034923  |25    |
|13823077 |25    |
|138230937|25    |
|4034761  |25    |
|4035131  |25    |
|13822847 |25    |
|138230911|25    |
|4034221  |25    |
+---------+------+
     */

    newDF
  }
}
