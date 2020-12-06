package cn.itcast.up.statistics

import cn.itcast.up.base.BaseModel
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * Author itcast
 * Date 2020/3/18 15:27
 * Desc 完成统计型标签-支付方式标签/模型的开发
 * 统计用户最常用的支付方式,然后给用户打上相应的标签
 * 如: 1用户,用了支付宝3次,  微信支付10次, 那么给用户打上支付方式为:微信支付对应的tagId
 */
object PaymentModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  //29	支付方式		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_orders##family=detail##selectFields=memberId,paymentCode		4	3
  override def getTagId(): Long = 29

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    //hbaseDF.show(false)
    //hbaseDF.printSchema()

    //fiveDS.show(false)
    //fiveDS.printSchema()

    /*
 +---------+-----------+
|memberId |paymentCode|
+---------+-----------+
|13823431 |alipay     |
|4035167  |alipay     |
|4035291  |alipay     |
|4035041  |alipay     |
|13823285 |kjtpay     |
|13823231 |cod        |
|47       |alipay     |
|13822713 |cod        |
|4033693  |cod        |
|13822859 |alipay     |
|13823431 |alipay     |
|13823587 |wspay      |
+---------+-----------+

root
 |-- memberId: string (nullable = true)
 |-- paymentCode: string (nullable = true)

+---+--------+
|id |rule    |
+---+--------+
|30 |alipay  |
|31 |wxpay   |
|32 |chinapay|
|33 |kjtpay  |
|34 |cod     |
|35 |other   |
+---+--------+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
     */

    //0.导入隐式转换
    import org.apache.spark.sql.functions._
    import spark.implicits._

    //1.根据用户id+支付方式进行分组并计数
    val tempDF: DataFrame = hbaseDF.groupBy('memberId, 'paymentCode)
      .agg(count('paymentCode) as "counts")
      .select('memberId as "userId", 'paymentCode, 'counts)
    tempDF.show(false)
    /*
 +--------+-----------+------+
|userId  |paymentCode|counts|
+--------+-----------+------+
|13823481|alipay     |96    |
|4035297 |alipay     |80    |
|13823317|kjtpay     |11    |
|13822857|alipay     |100   |
|4034541 |alipay     |96    |
|4034209 |cod        |17    |
|4034863 |alipay     |90    |
|4033371 |alipay     |95    |
|13822723|alipay     |148   |
|4034193 |cod        |16    |
|4035279 |cod        |18    |
     */

    //2.使用开窗函数进行组内排序,取Top1(每个用户使用最多的付款方式)
    //方式一:DSL风格开窗函数
    val tempDF2: Dataset[Row] = tempDF
      //withColumn("增加的这一列的列名",这一列如何计算得到的)
      //partitionBy的时候是按照用户id分,看每个用的个各个支付方式的使用次数排名
      .withColumn("rn", row_number().over(Window.partitionBy('userId).orderBy('counts.desc)))
    //tempDF2.show(false)
    /*
+---------+-----------+------+---+
|userId   |paymentCode|counts|rn |
+---------+-----------+------+---+
|13822725 |alipay     |89    |1  |
|13822725 |cod        |12    |2  |
|13822725 |kjtpay     |9     |3  |
|13822725 |wspay      |3     |4  |
|13822725 |giftCard   |2     |5  |
|13822725 |prepaid    |1     |6  |
|13823083 |alipay     |94    |1  |
|13823083 |cod        |18    |2  |
|13823083 |kjtpay     |12    |3  |
|13823083 |wspay      |7     |4  |
|13823083 |giftCard   |1     |5  |
|138230919|alipay     |98    |1  |
|138230919|cod        |15    |2  |
|138230919|kjtpay     |7     |3  |
|138230919|wspay      |3     |4  |
|138230919|chinapay   |1     |5  |
|138230919|chinaecpay |1     |6  |
|13823681 |alipay     |87    |1  |
|13823681 |cod        |11    |2  |
|13823681 |kjtpay     |5     |3  |
+---------+-----------+------+---+
only showing top 20 rows
     */

    val rankDF: Dataset[Row] = tempDF2.where('rn === 1)
    //rankDF.show(false)
    /*
+---------+-----------+------+---+
|userId   |paymentCode|counts|rn |
+---------+-----------+------+---+
|13822725 |alipay     |89    |1  |
|13823083 |alipay     |94    |1  |
|138230919|alipay     |98    |1  |
|13823681 |alipay     |87    |1  |
|4033473  |alipay     |113   |1  |
|13822841 |alipay     |86    |1  |
|13823153 |alipay     |102   |1  |
|13823431 |alipay     |99    |1  |
|4033348  |alipay     |112   |1  |
|4033483  |alipay     |84    |1  |
|4033575  |alipay     |101   |1  |
|4034191  |alipay     |87    |1  |
|4034923  |alipay     |84    |1  |
|13823077 |alipay     |104   |1  |
|138230937|alipay     |80    |1  |
|4034761  |alipay     |110   |1  |
|4035131  |alipay     |86    |1  |
|13822847 |alipay     |74    |1  |
|138230911|alipay     |101   |1  |
|4034221  |alipay     |86    |1  |
+---------+-----------+------+---+
only showing top 20 rows
     */

    //方式二:SQL风格开窗函数
    tempDF.createOrReplaceTempView("t_temp")
    val sql:String =
      """
        |select userId,paymentCode,counts,
        |row_number() over(partition by userId order by counts desc) rn
        |from t_temp
        |""".stripMargin
    val tempDF3: DataFrame = spark.sql(sql)
    tempDF3.show(false)
    val rankDF2: Dataset[Row] = tempDF3.where('rn === 1)
    rankDF2.show(false)
/*
+---------+-----------+------+---+
|userId   |paymentCode|counts|rn |
+---------+-----------+------+---+
|13822725 |alipay     |89    |1  |
|13822725 |cod        |12    |2  |
|13822725 |kjtpay     |9     |3  |
|13822725 |wspay      |3     |4  |
|13822725 |giftCard   |2     |5  |
|13822725 |prepaid    |1     |6  |
|13823083 |alipay     |94    |1  |
|13823083 |cod        |18    |2  |
|13823083 |kjtpay     |12    |3  |
|13823083 |wspay      |7     |4  |
|13823083 |giftCard   |1     |5  |
|138230919|alipay     |98    |1  |
|138230919|cod        |15    |2  |
|138230919|kjtpay     |7     |3  |
|138230919|wspay      |3     |4  |
|138230919|chinapay   |1     |5  |
|138230919|chinaecpay |1     |6  |
|13823681 |alipay     |87    |1  |
|13823681 |cod        |11    |2  |
|13823681 |kjtpay     |5     |3  |
+---------+-----------+------+---+
only showing top 20 rows

+---------+-----------+------+---+
|userId   |paymentCode|counts|rn |
+---------+-----------+------+---+
|13822725 |alipay     |89    |1  |
|13823083 |alipay     |94    |1  |
|138230919|alipay     |98    |1  |
|13823681 |alipay     |87    |1  |
|4033473  |alipay     |113   |1  |
|13822841 |alipay     |86    |1  |
|13823153 |alipay     |102   |1  |
|13823431 |alipay     |99    |1  |
|4033348  |alipay     |112   |1  |
|4033483  |alipay     |84    |1  |
|4033575  |alipay     |101   |1  |
|4034191  |alipay     |87    |1  |
|4034923  |alipay     |84    |1  |
|13823077 |alipay     |104   |1  |
|138230937|alipay     |80    |1  |
|4034761  |alipay     |110   |1  |
|4035131  |alipay     |86    |1  |
|13822847 |alipay     |74    |1  |
|138230911|alipay     |101   |1  |
|4034221  |alipay     |86    |1  |
+---------+-----------+------+---+
 */

    //3.将rankDF和fiveDS进行匹配
    //3.1将fiveDS转为map[支付方式,tagsId]
    //fiveDS.as[(tagsId,支付方式)]
    //fiveMap: Map[支付方式, tagsId]
    val fiveMap: Map[String, Long] = fiveDS.as[(Long, String)].map(t => {
      (t._2, t._1)
    }).collect.toMap

    //3.2将rankDF中的支付方式换成tagsId
    val paymentCode2tagsId = udf((paymentCode:String)=>{
      //如果支付方式获取到了直接返回对应的tagsId
      //如果支付方式没有获取到,那么返回-1,再判断如果是-1,用other去取对应的tagsId
      var tagsId: Long = fiveMap.getOrElse(paymentCode,-1)
      if(tagsId == -1){
        tagsId = fiveMap("other")
      }
      tagsId
    })
    val newDF: DataFrame = rankDF.select('userId,paymentCode2tagsId('paymentCode) as "tagsId")
    newDF.show(false)
    /*
+---------+------+
|userId   |tagsId|
+---------+------+
|13822725 |30    |
|13823083 |30    |
|138230919|30    |
|13823681 |30    |
|4033473  |30    |
|13822841 |30    |
     */

    //newDF
    null
  }
}
