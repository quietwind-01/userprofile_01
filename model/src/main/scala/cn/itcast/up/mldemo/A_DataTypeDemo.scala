package cn.itcast.up.mldemo

import org.apache.derby.impl.sql.execute.IndexRow
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Author itcast
 * Date 2020/3/20 9:36
 * Desc 演示SparkMLlib基础数据类型
 * LocalVector本地向量
 * LabelPoint标签向量
 * LocalMatrix本地矩阵
 * DistributedMatrix分布式矩阵
 */
object A_DataTypeDemo {
  def main(args: Array[String]): Unit = {
    println("===========TODO LocalVector本地向量=============")
    //TODO LocalVector本地向量
    //本地向量分为:
    //密集向量:(9,5,0,2,7)
    //稀疏向量:(不为0的个数4,不为零的位置(0,1,3,4),不为0的数据(9,5,2,7))
    val vector1: linalg.Vector = Vectors.dense(9,5,0,2,7)
    val vector2: linalg.Vector = Vectors.sparse(4,Array(0,1,3,4),Array(9,5,2,7))
    println(vector1)
    println(vector2)
    println(vector1(2))//0
    println(vector2(2))//0
    //[9.0,5.0,0.0,2.0,7.0]
    //(4,[0,1,3,4],[9.0,5.0,2.0,7.0])
    //0.0
    //0.0

    println("===========TODO LabelPoint标签向量=============")
    //TODO LabelPoint标签向量
    //LabelPoint标签向量表示带有标签的向,例如一个样本[标签,特征向量[特征1,特征2,特征3]]
    val labeledPoint1 = LabeledPoint(0,vector1)
    val labeledPoint2 = LabeledPoint(0,vector2)
    println(labeledPoint1)
    println(labeledPoint2)
    println(labeledPoint1.label)//获取标签:0
    println(labeledPoint1.features)//获取特征:[9.0,5.0,0.0,2.0,7.0]
    //(0.0,[9.0,5.0,0.0,2.0,7.0])
    //(0.0,(4,[0,1,3,4],[9.0,5.0,2.0,7.0]))
    //0.0
    //[9.0,5.0,0.0,2.0,7.0]

    println("===========TODO LocalMatrix本地矩阵=============")
    //TODO LocalMatrix本地矩阵
    //本地矩阵分为:
    //稠密矩阵:(几行,几列,(数据))
    //稀疏矩阵:(几行,几列,(位置),(位置),(数据))
    val matrix1: Matrix = Matrices.dense(3,2,Array(1,2,0,4,5,6))
    println(matrix1)
    //1.0  4.0
    //2.0  5.0
    //0.0  6.0
    // numRows: Int,几行
    // numCols: Int,几列
    // colPtrs: Array[Int],对应于新列开头的索引/对应列非零元素的个数1-0,3-1 == 1,2
    // rowIndices: Array[Int],条目的行索引/行向索引
    // values: Array[Double],数据
    //https://blog.csdn.net/qq595662096/article/details/89372898
    val matrix2: Matrix = Matrices.sparse(3,2,Array(0,1,3),Array(0,2,1),Array(9,6,8))
    println(matrix2)
    //3 x 2 CSCMatrix
    //(0,0) 9.0
    //(2,1) 6.0
    //(1,1) 8.0
    //还原一下:
    //9 0
    //0 8
    //0 6

    println("===========TODO DistributedMatrix分布式矩阵=============")
    //TODO DistributedMatrix分布式矩阵
    //分为://RowMatrix行矩阵//IndexedRowMatrix行索引矩阵//CoordinateMatrix三元组矩阵//BlockMatrix分块矩阵
    val sparkConf: SparkConf = new SparkConf().setMaster("local[*]").setAppName("ml")
    val sc = new SparkContext(sparkConf)
    sc.setLogLevel("WARN")
    val rdd: RDD[Array[Double]] = sc.parallelize(List(Array(1.0,2.0,3.0),Array(4.0,5.0,6.0)))
    val vectorRDD: RDD[linalg.Vector] = rdd.map(arr=>Vectors.dense(arr))
    println("=======RowMatrix行矩阵======")
    //RowMatrix行矩阵
    val rowMatrix = new RowMatrix(vectorRDD)
    rowMatrix.rows.foreach(println)
    //=======RowMatrix行矩阵======
    //[1.0,2.0,3.0]
    //[4.0,5.0,6.0]
    println("=======IndexedRowMatrix行索引矩阵======")
    //IndexedRowMatrix行索引矩阵
    val indexedRowRDD: RDD[IndexedRow] = vectorRDD.map(v => {
      new IndexedRow(v.size, v)
    })
    val indexedRowMatrix = new IndexedRowMatrix(indexedRowRDD)
    indexedRowMatrix.rows.foreach(println)
    //=======IndexedRowMatrix行索引矩阵======
    //IndexedRow(3,[1.0,2.0,3.0])
    //IndexedRow(3,[4.0,5.0,6.0])
  }
}
