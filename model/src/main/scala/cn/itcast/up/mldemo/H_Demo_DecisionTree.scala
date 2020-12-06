package cn.itcast.up.mldemo

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StandardScaler, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Author itcast
 * Date 2020/3/26 15:38
 * Desc 演示使用决策树算法完成鸢尾花数据集分类
 */
object H_Demo_DecisionTree {
  def main(args: Array[String]): Unit = {
    //0.准备环境和数据
    val spark: SparkSession = SparkSession.builder()
      .appName("ml")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    val csvDF: DataFrame = spark.read.csv("file:///D:\\授课\\191021-35\\用户画像\\day07\\data\\ml\\iris_DecisionTree.csv")
    val irisStringDF: DataFrame = csvDF.toDF("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species")
    val irisDF: DataFrame = irisStringDF.select(
      'Sepal_Length cast DoubleType,
      'Sepal_Width cast DoubleType,
      'Petal_Length cast DoubleType,
      'Petal_Width cast DoubleType,
      'Species
    )
    irisDF.show(10,false)
    /*
 +------------+-----------+------------+-----------+-----------+
|Sepal_Length|Sepal_Width|Petal_Length|Petal_Width|Species    |
+------------+-----------+------------+-----------+-----------+
|5.1         |3.5        |1.4         |0.2        |Iris-setosa|
|4.9         |3.0        |1.4         |0.2        |Iris-setosa|
|4.7         |3.2        |1.3         |0.2        |Iris-setosa|
通过观察上面的数据,发现数据是原始的格式,没有做其他的处理,后续需要做特征工程
     */

    //1.特征工程-标签数值化(使用StringIndexer,后续可以使用IndexToString还原)
    val stringIndexer: StringIndexer =  new StringIndexer()
      .setInputCol("Species")//指定对一列的字符串进行数值化
      .setOutputCol("Species_Indexer")//指定数值化之后该列的列名
    val stringIndexerModel: StringIndexerModel = stringIndexer.fit(irisDF)
    //stringIndexerModel.transform()
    //注意:后续使用SparkMLlib提供的Pipeline来构建整个的机器学习流程,所以这里就不需要再调用transform
    //可以理解为在Pipeline管道的最后传入数据时会自动的调用
    //当然如果先看一下也可以
    //stringIndexerModel.transform(irisDF).show()

    //2.特征工程-特征向量化
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"))
      .setOutputCol("features")


    //3.数据归一化
    val standaScaler: StandardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("StandardScaler_features")
      .setWithMean(true) //缩放到均值为0,标准差为1的正态分布
      .setWithStd(true)

    //4.构建决策树分类器
    val decisionTreeClassifier: DecisionTreeClassifier =  new DecisionTreeClassifier()
      .setFeaturesCol("StandardScaler_features")//设置特征列
      .setLabelCol("Species_Indexer")//设置标签列为数值化之后的列
      .setPredictionCol("predict")//设置预测列的名称(值为数字)
      //.setImpurity("gini")//设置不纯度/不确定性的衡量标准为基尼系数gini(默认)或信息熵entropy
      //.setMaxDepth(5)//设置树的最大深度(默认就是5),防止过拟合

    //5.还原标签列
    val indexToString: IndexToString =  new IndexToString()
      .setInputCol("predict")//对预测的结果(数字形式)进行还原
      .setOutputCol("predict_String")
      .setLabels(stringIndexerModel.labels)//原来数字和字符串的对应关系

    //6.数据集划分为80%训练集,20%测试集
    val Array(trainSet,testSet) = irisDF.randomSplit(Array(0.8,0.2),100)

    //7.构建Pipeline管道将上述的复杂的机器学习流程串联起来获得Pipeline管道对象
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(stringIndexerModel, vectorAssembler, standaScaler, decisionTreeClassifier, indexToString))

    //8.使用训练集训练Pipeline对象中的决策树模型
    val pipelineModel: PipelineModel = pipeline.fit(trainSet)

    //9.使用测试集进行测试预测结果
    val result: DataFrame = pipelineModel.transform(testSet)
    //val result2: DataFrame = pipelineModel.transform(trainSet)
    result.show(false)
    /*
-+---------------+---------------+-----------------+-------+---------------+
|Species        |Species_Indexer|features         |predict|predict_String |
+---------------+---------------+-----------------+-------+---------------+
|Iris-setosa    |0.0            |[4.5,2.3,1.3,0.3]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[4.6,3.4,1.4,0.3]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[4.6,3.6,1.0,0.2]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[4.7,3.2,1.3,0.2]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[4.8,3.4,1.9,0.2]|2.0    |Iris-virginica |预测错了
|Iris-setosa    |0.0            |[4.9,3.1,1.5,0.1]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[4.9,3.1,1.5,0.1]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[5.0,3.2,1.2,0.2]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[5.0,3.3,1.4,0.2]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[5.1,3.4,1.5,0.2]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[5.1,3.5,1.4,0.2]|0.0    |Iris-setosa    |
|Iris-setosa    |0.0            |[5.1,3.8,1.9,0.4]|1.0    |Iris-versicolor|预测错了
|Iris-setosa    |0.0            |[5.4,3.7,1.5,0.2]|0.0    |Iris-setosa    |
|Iris-versicolor|1.0            |[5.5,2.3,4.0,1.3]|1.0    |Iris-versicolor|
|Iris-setosa    |0.0            |[5.5,3.5,1.3,0.2]|0.0    |Iris-setosa    |
|Iris-versicolor|1.0            |[5.6,2.9,3.6,1.3]|1.0    |Iris-versicolor|
|Iris-setosa    |0.0            |[5.7,4.4,1.5,0.4]|0.0    |Iris-setosa    |
|Iris-versicolor|1.0            |[5.8,2.7,3.9,1.2]|1.0    |Iris-versicolor|
|Iris-virginica |2.0            |[6.0,2.2,5.0,1.5]|1.0    |Iris-versicolor|预测错了
|Iris-virginica |2.0            |[6.0,3.0,4.8,1.8]|2.0    |Iris-virginica |
+---------------+---------------+-----------------+-------+---------------+
only showing top 20 rows
     */

    //10.查看决策过程(查看决策树长啥样)
    //之前给Pipeline的是DecisionTreeClassifier决策树分类器,
    //经过Pipeline的训练之后形成了DecisionTreeClassificationModel决策树分类模型
    val decisionTreeClassificationModel: DecisionTreeClassificationModel = pipelineModel.stages(3).asInstanceOf[DecisionTreeClassificationModel]
    //decisionTreeClassificationModel就是训练好的决策树模型,里面就应该存储了决策过程
    val decisionTreestring: String = decisionTreeClassificationModel.toDebugString
    println("决策树的决策过程如下:\n"+decisionTreestring)
    /*
  决策树的决策过程如下:
DecisionTreeClassificationModel (uid=dtc_59145e873568) of depth 5 with 17 nodes
  If (feature 2 <= -1.285318889404062) //这里把feature2作为第一个决策依据的原因是什么?因为feature2的信息增益最大
   Predict: 0.0
  Else (feature 2 > -1.285318889404062)
   If (feature 3 <= 0.5777145152904802)
    If (feature 2 <= 0.6375608998063961)
     If (feature 0 <= -1.19533512343325)
      If (feature 1 <= -1.5236786563822888)
       Predict: 1.0
      Else (feature 1 > -1.5236786563822888)
       Predict: 2.0
     Else (feature 0 > -1.19533512343325)
      Predict: 1.0
    Else (feature 2 > 0.6375608998063961)
     If (feature 0 <= 0.12282892245805102)
      Predict: 1.0
     Else (feature 0 > 0.12282892245805102)
      Predict: 2.0
   Else (feature 3 > 0.5777145152904802)
    If (feature 2 <= 0.5210227307633379)
     If (feature 0 <= 0.0029958273770240252)
      Predict: 1.0
     Else (feature 0 > 0.0029958273770240252)
      Predict: 2.0
    Else (feature 2 > 0.5210227307633379)
     Predict: 2.0
     */

    //11.评估模型的好坏(查看模型的正确率和错误率)
      //创建多分类评估器
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      //让评估器对数据的预测类别和原来的类别进行对比,统计出评估结果
      .setLabelCol("Species_Indexer")//设置标签列,应该为数值化之后的标签列
      .setPredictionCol("predict")//设置预测列
      .setMetricName("accuracy")//设置评估标准为准确性/正确率
    //计算模型在测试集上的正确率和错误率
    val accuracy: Double = evaluator.evaluate(result)
    //val accuracy2: Double = evaluator.evaluate(result2)
    println("模型在测试集上的正确率为:"+accuracy)
    println("模型在测试集上的错误率为:"+(1-accuracy))
    //模型在测试集上的正确率为:0.9
    //模型在测试集上的错误率为:0.09999999999999998

    //println("模型在训练集上的正确率为:"+accuracy2)
    //println("模型在训练集上的错误率为:"+(1-accuracy2))
    //模型在训练集上的正确率为:1.0
    //模型在训练集上的错误率为:0.0
  }
}
