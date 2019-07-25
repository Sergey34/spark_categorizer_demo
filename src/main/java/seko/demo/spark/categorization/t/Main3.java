package seko.demo.spark.categorization.t;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StringType;
import scala.collection.JavaConverters;
import scala.collection.Seq;
import scala.collection.mutable.WrappedArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

@SuppressWarnings("Duplicates")
public class Main3 {
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


        List<Column> inputList = Arrays.asList(
                new Column("_source.full_stack_trace"),
                new Column("_source.exception_class"),
                new Column("_source.error_type"),
                new Column("_source.error_code"));
        Dataset<Row> cleanText = rawData.select(convertListToSeq(inputList))
                .na().fill("")
                .filter(new Column("_source.full_stack_trace").isNotNull())
                .withColumn("full_stack_trace_split", split(new Column("full_stack_trace"), "\n"))
                .withColumn("exception_class", split(new Column("exception_class"), "\n"))
                .withColumn("error_type", split(new Column("error_type"), "\n"))
                .withColumn("error_code", split(new Column("error_code"), "\n"))
                .withColumn("all_strings", concat(
                        new Column("full_stack_trace_split"),
                        new Column("exception_class"),
                        new Column("error_type"),
                        new Column("error_code")));
        cleanText.show(10);


        CountVectorizer cv = new CountVectorizer()
                .setInputCol("all_strings")
                .setOutputCol("index_all_strings")
                .setVocabSize(1000);

        Dataset<Row> ctCleanText = cv.fit(cleanText).transform(cleanText);


        IDF idf = new IDF().setInputCol("index_all_strings")
                .setOutputCol("idf_index_all_strings");

        Dataset<Row> transform = idf.fit(ctCleanText).transform(ctCleanText);
        transform.show();

        PipelineStage[] pipelineStages = {cv, idf};
        Pipeline pipeline = new Pipeline().setStages(pipelineStages);
        PipelineModel model = pipeline.fit(cleanText);
        Dataset<Row> ds = model.transform(cleanText);
        ds.show();

        String[] vocabulary = new String[0];
        for (Transformer stage : model.stages()) {
            if (stage instanceof CountVectorizerModel){
                vocabulary = ((CountVectorizerModel) stage).vocabulary();
            }
        }
        Broadcast<String[]> vocab_broadcast = sc.broadcast(vocabulary);


        LDA lda = new LDA().setK(25)
                .setSeed(321)
                .setOptimizer("em")
                .setFeaturesCol("idf_index_all_strings");

        LDAModel ldamodel = lda.fit(ds);

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

        datopicsMapped.show(100);
        datopicsMapped
                .withColumn("termIndices", datopicsMapped.col("termIndices").cast(DataTypes.StringType))
                .withColumn("termWeights", datopicsMapped.col("termWeights").cast(DataTypes.StringType))
                .withColumn("topic_desc", datopicsMapped.col("topic_desc").cast(DataTypes.StringType))
                .coalesce(1)
                .write().format("csv").csv("result");


    }
}
