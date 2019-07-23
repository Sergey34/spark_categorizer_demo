package seko.demo.spark.categorization

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.Column
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.split
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.types.StringType
import scala.collection.JavaConverters
import scala.collection.Seq
import scala.collection.mutable.WrappedArray
import java.io.Serializable
import java.util.*


class CategorizeService : Serializable {
    fun convertListToSeq(inputList: List<Column>): Seq<Column> {
        return JavaConverters.asScalaIteratorConverter(inputList.iterator()).asScala().toSeq()
    }

    fun categorize() {
        val sparkConf = SparkConf()
        sparkConf.setAppName("ProductRecommendation")
        sparkConf.setMaster("local[*]")
        val sc = JavaSparkContext(sparkConf)

        val spark = SQLContext(sc)

        val rawData = spark.read()
            .option("multiLine", true)
            .option("mode", "PERMISSIVE")
            .json("t.json")


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

        val lda = LDA().setK(25)
            .setSeed(321)
            .setOptimizer("em")
            .setFeaturesCol("features")


        val ldamodel = lda.fit(rescaledData)

        ldamodel.isDistributed()
        ldamodel.vocabSize()

        val ldatopics = ldamodel.describeTopics()
        ldatopics.show(25)


        val udfMapTermIdToWord = udf(object : Function1<WrappedArray.ofRef<Int>, List<String>> {
            override operator fun invoke(termIndices: WrappedArray.ofRef<Int>): List<String> {
                val words = ArrayList<String>()
                for (termIndex in termIndices.array()) {
                    words.add(vocab_broadcast.value()[termIndex as Int])
                }
                return words
            }

        }, ArrayType(StringType() as DataType, true) as DataType)
        val datopicsMapped = ldatopics.withColumn(
            "topic_desc",
            udfMapTermIdToWord.apply(convertListToSeq(listOf(ldatopics.col("termIndices"))))
        )

        datopicsMapped.show()
    }

}