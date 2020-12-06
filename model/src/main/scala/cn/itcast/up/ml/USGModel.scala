package cn.itcast.up.ml

import cn.itcast.up.base.BaseModel
import cn.itcast.up.bean.HBaseMeta
import cn.itcast.up.ml.PSMModel.spark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel, VectorAssembler, VectorIndexer}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, functions}
import spire.std.string

/**
 * Author itcast
 * Date 2020/3/27 9:40
 * Desc 使用决策树算法构建用户购物性别模型
 */
object USGModel extends BaseModel {
  def main(args: Array[String]): Unit = {
    execute()
  }

  //56	购物性别		inType=HBase##zkHosts=192.168.10.20##zkPort=2181##hbaseTable=tbl_goods##family=detail##selectFields=cOrderSn,ogColor,productType		4	36
  override def getTagId(): Long = 56

  override def compute(hbaseDF: DataFrame, fiveDS: Dataset[Row]): DataFrame = {
    hbaseDF.show(false)
    hbaseDF.printSchema()
    fiveDS.show(false)
    fiveDS.printSchema()
    /*
 从下面的tbl_goods(订单项表)数据可以发现:该表其实是一个订单项表,表示某订单下有哪些商品
 而一个用户可以有多个订单,所以还需要用户id和订单id的对应信息,也就是还需要从订单表中查询:
tbl_order(订单表--表示某用户有哪些订单)
 用户id 订单id

tbl_goods(订单项表--表示某订单下有哪些商品)
    订单id           商品颜色    商品类型
 +----------------------+---------+-----------+
|cOrderSn              |ogColor  |productType|
+----------------------+---------+-----------+
|jd_14091818005983607  |白色       |烤箱         |
|jd_14091818005983607  |香槟金      |冰吧         |
|jd_14092012560709235  |香槟金色     |净水机        |
|rrs_15234137          |梦境极光【布朗灰】|烤箱         |
|suning_790750687478116|梦境极光【卡其金】|4K电视       |
|rsq_805093707860210   |黑色       |烟灶套系       |
|jd_14090910361908941  |黑色       |智能电视       |
|jd_14091823464864679  |香槟金色     |燃气灶        |
|jd_14091817311906413  |银色       |滤芯         |
|suning_804226647488814|玫瑰金      |电饭煲        |
|amazon_810093307599049|蓝色       |Leader/统帅冰箱|
|amazon_793590614501672|粉色       |烤箱         |
|suning_799355971458879|黑色       |净水机        |
|amazon_796646707477257|香槟色      |智能电视       |
|suning_804533210244923|蓝色       |烤箱         |
|jd_14090709431827457  |粉色       |挂烫机        |
|suning_794987009664150|银色       |净水机        |
|amazon_795269235907650|黑色       |其他         |
|140916959656659       |月光银      |Haier/海尔冰箱 |
|suning_794588452960075|金色       |烤箱         |
+----------------------+---------+-----------+
only showing top 20 rows

root
 |-- cOrderSn: string (nullable = true)
 |-- ogColor: string (nullable = true)
 |-- productType: string (nullable = true)

+---+----+
|id |rule|
+---+----+
|57 |0   |男
|58 |1   |女
|59 |-1  |中性
+---+----+

root
 |-- id: long (nullable = false)
 |-- rule: string (nullable = true)
     */

    //0.导入隐式转换
    import spark.implicits._
    import scala.collection.JavaConversions._
    import org.apache.spark.sql.functions._

    //0.额外查询订单表
    val orderDF: DataFrame = spark.read.format("cn.itcast.up.tools.HBaseSource")
      .option(HBaseMeta.INTYPE, "HBase")
      .option(HBaseMeta.ZKHOSTS, "192.168.10.20")
      .option(HBaseMeta.ZKPORT, "2181")
      .option(HBaseMeta.HBASETABLE, "tbl_orders")
      .option(HBaseMeta.FAMILY, "detail")
      .option(HBaseMeta.SELECTFIELDS, "memberId,orderSn")
      .load()

    //1.将订单表orderDF订单项表hbaseDF进行关联获取得到用户购买了哪些商品:
    //用户id、商品特征
    val userGoodsDF: DataFrame = hbaseDF.join(orderDF, 'cOrderSn === 'orderSn)
      .select('memberId as "userId", 'ogColor, 'productType)
    userGoodsDF.show(false)
    /*
 用户id     商品颜色   商品类型
+---------+---------+-----------+
|userId   |ogColor  |productType|gender标注的购物性别
+---------+---------+-----------+
|13823535 |黑色       |游戏鼠标   |男
|13823535 |粉色       |女式皮包   |女
|13823535 |粉色       |燃气热水器 |
|13823391 |乐享金      |冰吧     |
|4034493  |金色       |LED电视   |
|13823683 |金属灰      |前置过滤器|
|62       |金色       |冷柜      |
+---------+---------+-----------+

上面的数据就是用户的历史购物记录,我们要使用决策树算法从中发掘规律,给用户打上购物性别
但是需要注意:决策树算法是一个有监督学习分类算法
所以还需要给数据做好数据标注,数据标注在实际中由人工手动标注
那么问题来了:数据都标注好了男女,还有用算法干嘛?
注意:人工手动标注工作量太大,所以应该使用这一部分标准好的数据训练出一个模型,
那么未来对于新的数据就可以使用训练好的模型来直接预测该用户的购物性别!
     */

    //注意:我们的目标是使用决策树算法构建用户购物性别模型
    //但是决策数是一个有监督的分类算法
    //需要数据提供特征(从哪些角度判断/决策用户购物性别是男还是女)和标签(标注好的用户购物性别是男还是女)
    //2.数据标注(实际中是人工手动的标注!!!我们这里用程序模拟一下即可)
    //我们这里人工/程序/算法判断用户购物性别是男/女,使用的特征较少只有商品颜色和商品类型,是因为数据量不够,如果太多的话运行时间也太长
    //注意:实际中会使用更多的特征训练用户购物性别模型,如商品颜色,类型,尺寸,包装,品牌,代言人...
    val label: Column = functions.when('ogColor.equalTo("樱花粉")
      .or('ogColor.equalTo("白色"))
      .or('ogColor.equalTo("香槟色"))
      .or('ogColor.equalTo("香槟金"))
      .or('productType.equalTo("料理机"))
      .or('productType.equalTo("挂烫机"))
      .or('productType.equalTo("吸尘器/除螨仪")), "女") //女
      .otherwise("男") //男
      .alias("gender") //标记的列名

    //val userGoodsLabelDF: DataFrame = userGoodsDF.select('userId, 'ogColor, 'productType,label)
    //userGoodsLabelDF.show(false)
    /*
                                人工标注的男/女
+---------+---------+-----------+------+
|userId   |ogColor  |productType|gender|
+---------+---------+-----------+------+
|13823535 |灰色       |其他     |男     |
|13823535 |银色       |智能电视 |男     |
|91       |香槟金      |其他    |女
     */
    //上面的步骤走完之后,相等于数据集准备好了,已经标注好购物性别标签了

    //将商品颜色和商品类型的字符串形式转为数值形式
    //而一般商品颜色和商品类型的字符串形式对应的数值形式在数据库的字典表中有
    //这里就直接给商品颜色和商品类型赋上数值(相当于从数据库中查询再赋上)
    val color: Column = functions
      .when('ogColor.equalTo("银色"), 1)
      .when('ogColor.equalTo("香槟金色"), 2)
      .when('ogColor.equalTo("黑色"), 3)
      .when('ogColor.equalTo("白色"), 4)
      .when('ogColor.equalTo("梦境极光【卡其金】"), 5)
      .when('ogColor.equalTo("梦境极光【布朗灰】"), 6)
      .when('ogColor.equalTo("粉色"), 7)
      .when('ogColor.equalTo("金属灰"), 8)
      .when('ogColor.equalTo("金色"), 9)
      .when('ogColor.equalTo("乐享金"), 10)
      .when('ogColor.equalTo("布鲁钢"), 11)
      .when('ogColor.equalTo("月光银"), 12)
      .when('ogColor.equalTo("时尚光谱【浅金棕】"), 13)
      .when('ogColor.equalTo("香槟色"), 14)
      .when('ogColor.equalTo("香槟金"), 15)
      .when('ogColor.equalTo("灰色"), 16)
      .when('ogColor.equalTo("樱花粉"), 17)
      .when('ogColor.equalTo("蓝色"), 18)
      .when('ogColor.equalTo("金属银"), 19)
      .when('ogColor.equalTo("玫瑰金"), 20)
      .otherwise(0)
      .alias("color")
    //类型ID应该来源于字典表,这里简化处理
    val productType: Column = functions
      .when('productType.equalTo("4K电视"), 9)
      .when('productType.equalTo("Haier/海尔冰箱"), 10)
      .when('productType.equalTo("Haier/海尔冰箱"), 11)
      .when('productType.equalTo("LED电视"), 12)
      .when('productType.equalTo("Leader/统帅冰箱"), 13)
      .when('productType.equalTo("冰吧"), 14)
      .when('productType.equalTo("冷柜"), 15)
      .when('productType.equalTo("净水机"), 16)
      .when('productType.equalTo("前置过滤器"), 17)
      .when('productType.equalTo("取暖电器"), 18)
      .when('productType.equalTo("吸尘器/除螨仪"), 19)
      .when('productType.equalTo("嵌入式厨电"), 20)
      .when('productType.equalTo("微波炉"), 21)
      .when('productType.equalTo("挂烫机"), 22)
      .when('productType.equalTo("料理机"), 23)
      .when('productType.equalTo("智能电视"), 24)
      .when('productType.equalTo("波轮洗衣机"), 25)
      .when('productType.equalTo("滤芯"), 26)
      .when('productType.equalTo("烟灶套系"), 27)
      .when('productType.equalTo("烤箱"), 28)
      .when('productType.equalTo("燃气灶"), 29)
      .when('productType.equalTo("燃气热水器"), 30)
      .when('productType.equalTo("电水壶/热水瓶"), 31)
      .when('productType.equalTo("电热水器"), 32)
      .when('productType.equalTo("电磁炉"), 33)
      .when('productType.equalTo("电风扇"), 34)
      .when('productType.equalTo("电饭煲"), 35)
      .when('productType.equalTo("破壁机"), 36)
      .when('productType.equalTo("空气净化器"), 37)
      .otherwise(0)
      .alias("productType")

    val dataDF: DataFrame = userGoodsDF.select('userId, color, productType, label)
    dataDF.show(false)
    /*
 +---------+-----+-----------+------+
|userId   |color|productType|gender|
+---------+-----+-----------+------+
|13823535 |16   |0          |男     |
|13823535 |1    |24         |男     |
|91       |15   |0          |女     |
+---------+-----+-----------+------+
     */

    //3.特征工程-类别标签数值化
    val stringIndexerModel: StringIndexerModel = new StringIndexer()
      .setInputCol("gender") //编码前
      .setOutputCol("gender_index") //编码后
      .fit(dataDF)
    //stringIndexerModel.transform(userGoodsDF).show()

    //4.特征工程-特征向量化-特征编码(前面的颜色/类型做的编码是数据字典表中对应的数字,有些是不太符合机器学习的规范的)
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("color", "productType"))
      .setOutputCol("features")

    val vectorIndexer: VectorIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("features_index")

    //5.构建决策树
    val decisionTreeClassifier: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setFeaturesCol("features_index")
      .setLabelCol("gender_index")
      .setPredictionCol("predict")
    //maxDepth：限定决策树的最大可能深度。
    //minInfoGain:最小信息增益（设置阈值），小于该值将不带继续分叉;
    //minInstancesPerNode：如果某个节点的样本数量小于该值，则该节点将不再被分叉。（设置阈值）
      //.setMaxDepth(5)
      //.setMinInfoGain(1.0)
      //.setMinInstancesPerNode(1)
      //.setImpurity("gini")

   /* val randomForestClassifier: RandomForestClassifier = new RandomForestClassifier()
      .setNumTrees(10)//设置随机森林中决策树的颗数
      .setFeaturesCol("features_index")
      .setLabelCol("gender_index")
      .setPredictionCol("predict")*/


    //6.还原标签列
    val indexToString: IndexToString = new IndexToString()
      .setInputCol("predict")
      .setOutputCol("predict_String")
      .setLabels(stringIndexerModel.labels)

    //7.划分数据集
    val Array(trainSet, testSet) = dataDF.randomSplit(Array(0.8, 0.2), 100)



    //8.构建Pipeline进行训练
    val pipelineModel: PipelineModel = new Pipeline().setStages(Array(stringIndexerModel, vectorAssembler, vectorIndexer, decisionTreeClassifier, indexToString))
      .fit(trainSet)

    //9.预测
    val testResult: DataFrame = pipelineModel.transform(testSet) //在测试集上结果
    val trainResult: DataFrame = pipelineModel.transform(trainSet) //在训练集上的结果

    //10.查看决策过程
    val decisionTreeClassificationModel: DecisionTreeClassificationModel = pipelineModel.stages(3).asInstanceOf[DecisionTreeClassificationModel]
    val debugString: String = decisionTreeClassificationModel.toDebugString
    println(debugString)

    //DecisionTreeClassificationModel类.load()
    //decisionTreeClassificationModel对象.save()

    //11.模型评估
   /* val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      //让评估器对数据的预测类别和原来的类别进行对比,统计出评估结果
      .setLabelCol("gender_index") //设置标签列,应该为数值化之后的标签列
      .setPredictionCol("predict") //设置预测列
      .setMetricName("f1") //设置评估标准为准确性/正确率
    //计算模型在测试集上的正确率和错误率
    val f1: Double = evaluator.evaluate(testResult)
    println("模型在测试集上f1Score为:" + f1)
    //println("模型在测试集上的错误率为:" + (1 - accuracy))*/
    evaluateF1AndAUC(trainResult,testResult)

    //12.查看结果
    val allResult: Dataset[Row] = testResult.union(trainResult)
    allResult.show(false)
    /*
+--------+-----+-----------+------+-------------+-------+--------------+
|userId  |color|productType|gender|gender_index|predict|predict_String|male|femal
+--------+-----+-----------+------+-------------+-------+--------------+
|91      |7    |10         |男     |0.0          |0.0    |男             |1    0
|91      |15   |0          |女     |1.0          |1.0    |女             |0    1
|91      |10   |10         |男     |0.0          |0.0    |男             |1    0
|13823317|15   |13         |女     |1.0          |1.0    |女             |
|13823363|20   |23         |女     |1.0          |1.0    |女             |
|4034793 |7    |0          |男     |0.0          |0.0    |男             |
|4034881 |1    |0          |男     |0.0          |0.0    |男             |
|4034949 |3    |19         |女     |1.0          |1.0    |女             |
|13823275|16   |0          |男     |0.0          |0.0    |男             |
|13822847|10   |30         |男     |0.0          |0.0    |男             |
|13823035|13   |23         |女     |1.0          |1.0    |女             |
+--------+-----+-----------+------+-------------+-------+--------------+
only showing top 20 rows

|userId  |male|femal|total
|91       |2    1     3

     */
    //后续步骤思路分析:
    //==1.目标:
    //通过上面的数据可以得出一个训练好的决策树模型
    //那么后续就可以使用该模型对新数据进行购物性别的预测
    //但是我们学习时没有额外的数据了,就假设上面的结果就是对新数据的预测结果
    //那么我们后面的任务就变成了要给这些用户打上性别标签

    //==2.数据
    //通过观察数据我们后续只需要获取如下格式的数据即可:
    //userId,predict
    //1      0男
    //2      1女
    //好像这样就可以和fiveDS进行关联,然后给用户打上tagsId了
    //但是注意:实际得到的数据是这样的:
    //userId,predict
    //1       0
    //1       1
    //1       1
    //  .....

    //==3.问题:
    //也就是同一个用户一般会买多个商品,也就是会被预测多次购物性别
    //而这预测的多次购物性别结果可能不一样啊!
    //那么最后到底该给这个用户打上什么标签呢?


    //==4.解决方案:
    //所以得统计一下该用户被预测为男多少次,预测为女多少次!,也就是要得出下面的数据
    //userId,male男,female女,total总共预测多少次
    //1      2次      8次     10
    //  ...
    //也就是说统计出各个用户被预测为男和女的次数,最后再决定到底给该用户打上什么标签:
    //userId,tagsId
    //1       女对应的标签id-58

    //==5.将解决方案转为代码实现:
    //13.给用打tagsId,得到userId,tagsId
    //13.1将userId,predict转为userId用户id,male预测为男的次数,female预测为女的次数,total总共预测的次数
    val predictCountDF: DataFrame = allResult.select(
      'userId,
      when('predict === 0.0, 1).otherwise(0).as("male"),
      when('predict === 1.0, 1).otherwise(0).as("female")
    ).groupBy('userId)
      .agg(
        sum('male).cast(DoubleType) as "male",
        sum('female).cast(DoubleType) as "female",
        count('userId).cast(DoubleType) as "total"
      )
    predictCountDF.show(false)
    //用户id, 预测为男的次数,女的次数,总共被预测的次数
    //userId,   male,      female,  total
    /*
    +---------+----+------+-----+
    |userId   |male|female|total|
    +---------+----+------+-----+
    |13823083 |13.0|4.0   |17.0 |
    |4033473  |12.0|1.0   |13.0 |
    |13823681 |3.0 |0.0   |3.0  |
    |138230919|3.0 |2.0   |5.0  |
    |13822725 |5.0 |2.0   |7.0  |
     */

    //13.2根据用户被预测的男女次数的比率决定用户到底被标记为哪一类购物性别标签
    //根据运营或者产品提供的规则:
    //预测为男的次数比率>60%,则购物性别标记为男对应的tagId
    //预测为女的次数比率>60%,则购物性别标记为女对应的tagId
    //上述都不满足,则购物性别标记为中性对应的tagId

    //13.2.1先准备男/女/中性对应的tagId的Map,也就是先处理fiveDS
    val gender2TagIdMap: Map[String, Long] = fiveDS.as[(Long, String)].map(t => {
      (t._2, t._1)
    }).collect().toMap

    //13.2.2将predictCountDF中的male|female|total使用UDF转为tagId
    val gender2TagIdsUDF = udf((male: Double, female: Double, total: Double) => {
      val maleRate = male / total //预测为男的次数的比率
      val femaleRate = female / total //预测为女的次数的比率
      if (maleRate > 0.6) {
        //预测为男的次数比率>60%,则购物性别标记为男对应的tagId
        gender2TagIdMap("0")
      } else if (femaleRate > 0.6) {
        //预测为女的次数比率>60%,则购物性别标记为女对应的tagId
        gender2TagIdMap("1")
      } else {
        //上述都不满足,则购物性别标记为中性对应的tagId
        gender2TagIdMap("-1")
      }
    })
    val newDF: DataFrame = predictCountDF.select('userId, gender2TagIdsUDF('male, 'female, 'total) as "tagIds")
    newDF.show(false)
    /*
  +---------+------+
|userId   |tagIds|
+---------+------+
|13823083 |57    |
|4033473  |57    |
|13823681 |57    |
|138230919|59    |
|13822725 |57    |
|4034923  |59    |
|13823431 |59    |
|13823077 |58    |
     */
    null
  }

  //提供工具类可以查看模型在训练集/测试集上的综合指标:F1和AUC
  def evaluateF1AndAUC(trainResultDF: DataFrame, testResultDF: DataFrame): Unit = {
    // 1. F1-综合查准率和召回率
    val accEvaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("predict")
      .setLabelCol("label")
      .setMetricName("f1")

    val trainF1: Double = accEvaluator.evaluate(trainResultDF)
    val testF1: Double = accEvaluator.evaluate(testResultDF)
    println(s"训练集上的 f1 是 : $trainF1")
    println(s"测试集上的 f1 是 : $testF1")

    // 2. AUC-ROC曲线下的面积-综合了查出率和查错率
    val trainRdd: RDD[(Double, Double)] = trainResultDF.select("gender_index", "predict").rdd
      .map(row => (row.getAs[Double](0), row.getAs[Double](1)))
    val testRdd: RDD[(Double, Double)] = testResultDF.select("gender_index", "predict").rdd
      .map(row => (row.getAs[Double](0), row.getAs[Double](1)))

    val trainAUC: Double = new BinaryClassificationMetrics(trainRdd).areaUnderROC()
    val testAUC: Double = new BinaryClassificationMetrics(testRdd).areaUnderROC()
    println(s"训练集上的 AUC 是 : $trainAUC")
    println(s"测试集上的 AUC 是 : $testAUC")
  }


}
