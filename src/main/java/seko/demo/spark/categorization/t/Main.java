package seko.demo.spark.categorization.t;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import scala.collection.JavaConverters;
import scala.collection.Seq;
import scala.collection.mutable.WrappedArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.split;
import static org.apache.spark.sql.functions.udf;

public class Main {
    public static Seq<Column> convertListToSeq(List<Column> inputList) {
        return JavaConverters.asScalaIteratorConverter(inputList.iterator()).asScala().toSeq();
    }

    public static void main(String[] args) {
        categorize();
    }

    public static void categorize() {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("ProductRecommendation");
        sparkConf.setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        SQLContext spark = new SQLContext(sc);

        Dataset<Row> rawData = spark.read()
                .option("multiLine", true)
                .option("mode", "PERMISSIVE")
                .json("t.json");


        List<Column> inputList = Arrays.asList(new Column("_source.full_stack_trace"));
        Dataset<Row> cleanText = rawData.select(convertListToSeq(inputList))
                .filter(new Column("_source.full_stack_trace").isNotNull())
                .withColumn("full_stack_trace_split", split(new Column("full_stack_trace"), "\n"));
        cleanText.show(10);
        cleanText.printSchema();

        CountVectorizer cv = new CountVectorizer()
                .setInputCol("full_stack_trace_split")
                .setOutputCol("full_stack_traceFeatures")
                .setVocabSize(1000);

        CountVectorizerModel cvmodel = cv.fit(cleanText);
        Dataset<Row> featurizedData = cvmodel.transform(cleanText);

        String[] vocab = cvmodel.vocabulary();

        Broadcast<String[]> vocab_broadcast = sc.broadcast(vocab);

        IDF idf = new IDF().setInputCol("full_stack_traceFeatures")
                .setOutputCol("features");


        IDFModel idfModel = idf.fit(featurizedData);
        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        rescaledData.show();

        LDA lda = new LDA().setK(25)
                .setSeed(321)
                .setOptimizer("em")
                .setFeaturesCol("features");


        LDAModel ldamodel = lda.fit(rescaledData);

        ldamodel.isDistributed();
        ldamodel.vocabSize();

        Dataset<Row> ldatopics = ldamodel.describeTopics();
        ldatopics.show(25);


        UserDefinedFunction udfMapTermIdToWord = udf((WrappedArray.ofRef termIndices) -> {
                    List<String> words = new ArrayList<>();
                    for (Object termIndex : termIndices.array()) {
                        words.add(vocab_broadcast.value()[(int) termIndex]);
                    }
                    return words;
                },
                DataTypes.createArrayType(DataTypes.StringType)
        );

        Dataset<Row> datopicsMapped = ldatopics.withColumn(
                "topic_desc",
                udfMapTermIdToWord.apply(convertListToSeq(Arrays.asList(ldatopics.col("termIndices"))))
        );

        datopicsMapped.show();
    }
}
