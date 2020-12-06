package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.common.HDFSUtils
import cn.itcast.up.tools.KMeansUtils
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}

import scala.collection.{immutable, mutable}

/**
 * Author itcast
 * Date 2020/3/24 10:04
 * Desc PSM(Price Sensitivity Measurement)价格敏感度模型
 * psm = 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
 * 有时在实际业务中，会把用户分为3-5类，
 * 比如分为价格极度敏感、较敏感、一般敏感、较不敏感、极度不敏感。
 * 然后将每类的聚类中心值与实际业务所需的其他指标结合，最终确定人群类别，判断在不同需求下是否触达或怎样触达。
 * 比如电商要通过满减优惠推广一新品牌的麦片，
 * 此时可优先选择优惠敏感且对麦片有消费偏好的用户进行精准推送，
 * 至于优惠敏感但日常对麦片无偏好的用户可暂时不进行推送或减小推送力度，
 * 优惠不敏感且对麦片无偏好的用户可选择不进行推送。
 * 可见，在实际操作中，技术指标评价外，还应结合业务需要，才能使模型达到理想效果。
 * 也就是说实际中价格敏感度不光光是单独的使用,还可以和其他模型组合使用
 * 那么对于其他模型也都是一样
 * 如,对北京地区,90后,男性青年,推送电子产品(用到了三个画像标签)
 */
object PSMModel extends BaseModel{
  def main(args: Array[String]): Unit = {
    execute()
  }
  //50	价格敏感度		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_orders##family=detail##selectFields=memberId,orderSn,orderAmount,couponCodeValue		4	36
  override def getTagId(): Long = 50

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    hbaseDF.show(false)
    hbaseDF.printSchema()
    fiveDS.show(false)
    fiveDS.printSchema()
    /*
用户id      订单id             订单实际金额     优惠价格
+---------+-------------------+-----------+---------------+
|memberId |orderSn            |orderAmount|couponCodeValue|state
+---------+-------------------+-----------+---------------+
|13823431 |ts_792756751164275 |2479.45    |10.00          | 1
|4035167  |D14090106121770839 |2449.00    |0.00           | 0
|4035291  |D14090112394810659 |1099.42    |0.00           |
|4035041  |fx_787749561729045 |1999.00    |0.00           |

root
 |-- memberId: string (nullable = true)
 |-- orderSn: string (nullable = true)
 |-- orderAmount: string (nullable = true)
 |-- couponCodeValue: string (nullable = true)

+---+----+
|id |rule|
+---+----+
|51 |1   |极度敏感
|52 |2   |比较敏感
|53 |3   |一般敏感
|54 |4   |不太敏感
|55 |5   |极度不敏感
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
     */
    //0.导入隐式转换
    import spark.implicits._
    import scala.collection.JavaConversions._
    import org.apache.spark.sql.functions._

    //0.定义字符串常量
    val psmScoreStr: String = "psm"
    val featureStr: String = "feature"
    val predictStr: String = "predict"


    //1.计算每个用户的PSM
    //PSM1 = 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
    //PSM2 = (优惠订单数/总订单数) + (平均优惠金额/平均每单应收) + (优惠总金额/应收总金额)
    //PSM3 = (优惠订单数/总订单数) + ((优惠总金额/优惠订单数)/(应收总金额/总订单数)) + (优惠总金额/应收总金额)
    //最终要计算
    //PSM1= 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
    //其实就是要计算
    //PSM3 = (优惠订单数/总订单数) + ((优惠总金额/优惠订单数)/(应收总金额/总订单数)) + (优惠总金额/应收总金额)
    //而最终计算每个用户的PSM3(PSM)只需要计算每个用户的如下4个指标即可：
    //优惠订单数、总订单数、优惠总金额、应收总金额
    //1.1为了计算上述指标,先计算下面的:
    //ra:receivableAmount应收金额 = 实际订单金额 + 优惠金额
    val raColumn:Column = 'orderAmount + 'couponCodeValue as "ra"
    //da:discountAmount折扣金额/优惠金额
    val daColume:Column = 'couponCodeValue as "da"
    //pa:practicalAmount实收金额
    val paColume:Column = 'orderAmount as "pa"
    //state:订单状态,优惠订单状态为1(优惠金额不等于0,则为1),非优惠订单状态为0
    val stateColume:Column = when('couponCodeValue =!= 0,1)
      .when('couponCodeValue === 0,0)
      .as("state")

    //1.2执行查询计算用户的"ra","da","pa","state"
    val tempDF: DataFrame = hbaseDF.select('memberId as "userId",raColumn,daColume,paColume,stateColume)
    tempDF.show(false)
    /*
 用户id   应收金额  优惠金额 实收金额 订单状态
 +---------+-------+------+-------+-----+
|userId   |ra     |da    |pa     |state|
+---------+-------+------+-------+-----+
|13823431 |2479.45|0.00  |2479.45|0    |
|4035167  |2449.0 |0.00  |2449.00|0    |
|13823431 |1899.0 |200.00|1699.00|1    |
|13823587 |1558.0 |0.00  |1558.00|0    |
+---------+-------+------+-------+-----+
only showing top 20 rows
     */
    //1.3计算每个用户的优惠订单数、总订单数、优惠总金额、应收总金额
    //tdon:total discount order num优惠订单数
    val tdonColumn:Column = sum('state) as "tdon"
    //ton:total order num总订单数
    val tonColumn:Column = count('state) as "ton"
    //tda:total discount amount优惠总金额
    val tdaColumn:Column = sum('da) as "tda"
    //tra:total receivable amount应收总金额
    val traColumn:Column = sum('ra) as "tra"
    val tempDF2: DataFrame = tempDF.groupBy('userId)
      .agg(tdonColumn, tonColumn, tdaColumn, traColumn)
    tempDF2.show(false)
    /*
  用户id   优惠订单数、总订单数、优惠总金额、应收总金额
 +---------+----------+-------+------------+------------------+
|userId   |tdon       |ton     |tda        |tra               |
+---------+-----------+-------+------------+------------------+
|4033473  |3          |142     |500.0      |252430.92         |
|13822725 |4          |116     |800.0      |180098.34         |
|13823681 |1          |108     |200.0      |169946.1          |
|138230919|3          |125     |600.0      |240661.56999999998|
|13823083 |3          |132     |600.0      |234124.17         |
     */
    //1.4计算每个用的PSM
    //PSM1 = 优惠订单占比 + 平均优惠金额占比 + 优惠总金额占比
    //PSM2 = (优惠订单数/总订单数) + (平均优惠金额/平均每单应收) + (优惠总金额/应收总金额)
    //PSM3 = (优惠订单数/总订单数) + ((优惠总金额/优惠订单数)/(应收总金额/总订单数)) + (优惠总金额/应收总金额)
    val psmColumn:Column = ('tdon/'ton) + (('tda/'tdon)/('tra/'ton)) + ('tda/'tra)  as psmScoreStr
    val PSMDF: Dataset[Row] = tempDF2.select('userId, psmColumn)
      //上面计算psm的过程中有很多除法,tdon优惠订单数分母为0的话,SparkSQL会返回null,所以可以过滤掉null
      //但是实际中,tdon优惠订单数分母为0应该表示用户对于价格不敏感,计算时可以给个很小的值如1
      .filter('psm.isNotNull)
    PSMDF.show(false)
    /*
 +---------+-------------------+
|userId   |psm                |
+---------+-------------------+
|4033473  |0.11686252330855691|
|13822725 |0.16774328728519597|
|13823681 |0.13753522440350205|
     */

    //2.使用KMeans算法聚类
    //===========================================================
    //有些步骤可以封装抽取,思考下如何封装抽取
    //1.特征向量化
    val inputCols: Array[String] = Array("psm")
    val vectorDF: DataFrame = KMeansUtils.getVecotrDF(PSMDF,inputCols)
    //vectorDF.show(false)
    //vectorDF.printSchema()

    //2.选取K值
    println("k值选取开始")
    val ks:List[Int] = List(2,3,4,5,6,7,8,9)
    val k: Int = KMeansUtils.selectK(vectorDF, ks)
    println("k值选取结束")
    //假设上面的执行了,选取出的k为:5

    //3.模型加载或训练再保存
    val path: String = "/model/PSM35"
    val model: KMeansModel = KMeansUtils.getKMeansModel(path,k,20,vectorDF)

    //4.预测
    val predictResultDF: DataFrame = model.transform(vectorDF)
    predictResultDF.show(false)

    //5.聚类中心编号和tagsId相对应
    val indexAndTagsIdMap: Map[Int, Long] = KMeansUtils.getIndexAndTagsIdMap(model,fiveDS)

    //6.获取userId和tagsId
    val newDF: DataFrame = KMeansUtils.getNewDF(predictResultDF,indexAndTagsIdMap)
    newDF.show(false)
    /*
 +---------+------+
|userId   |tagsId|
+---------+------+
|4033473  |54    |
|13822725 |53    |
|13823681 |53    |
|138230919|53    |
|13823083 |53    |
     */


    null
  }
}
