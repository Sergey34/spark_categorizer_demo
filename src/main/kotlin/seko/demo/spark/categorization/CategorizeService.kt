package seko.demo.spark.categorization

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.RowFactory
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.split
import scala.collection.JavaConverters
import scala.collection.Seq
import java.io.Serializable


class CategorizeService : Serializable {
    fun convertListToSeq(inputList: List<Column>): Seq<Column> {
        return JavaConverters.asScalaIteratorConverter(inputList.iterator()).asScala().toSeq()
    }

    @Throws(Exception::class)
    fun call(arg0: String): Row {
        // TODO Auto-generated method stub
        return RowFactory.create(arg0.split(",".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()[0], "1,2,3,4")
    }

    fun categorize() {
        val sparkConf = SparkConf()
        sparkConf.setAppName("ProductRecommendation")
        sparkConf.setMaster("local[*]")
        val sc = JavaSparkContext(sparkConf)

        val spark = SQLContext(sc)

        var rawData = spark.read()
            .option("multiLine", true)
            .option("mode", "PERMISSIVE")
            .json("tt.json")


        val inputList = listOf(Column("_source.full_stack_trace"))
        val cleanText = rawData.select(convertListToSeq(inputList))
            .filter(Column("_source.full_stack_trace").isNotNull)
            .withColumn("full_stack_trace_split", split(Column("full_stack_trace"), "\n"))
        cleanText.show(10)
        cleanText.printSchema()

        val cv = CountVectorizer()
            .setInputCol("full_stack_trace_split")
            .setOutputCol("full_stack_traceFeatures")
            .setVocabSize(1000)

        val cvmodel = cv.fit(cleanText)
        val featurizedData = cvmodel.transform(cleanText)

        val vocab = cvmodel.vocabulary()
        val vocab_broadcast = sc.broadcast(vocab)

        val idf = IDF().setInputCol("full_stack_traceFeatures")
            .setOutputCol("features")


        val idfModel = idf.fit(featurizedData)
        val rescaledData = idfModel.transform(featurizedData)
        rescaledData.show()
        println("asda")
        val lda = LDA().setK(25)
            .setSeed(321)
            .setOptimizer("em")
            .setFeaturesCol("features")


        val ldamodel = lda.fit(rescaledData)

        ldamodel.isDistributed()
        ldamodel.vocabSize()

        val ldatopics = ldamodel.describeTopics()
        ldatopics.show(25)

    }

}