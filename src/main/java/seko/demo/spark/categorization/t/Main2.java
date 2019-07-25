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
import scala.collection.JavaConverters;
import scala.collection.Seq;
import scala.collection.mutable.WrappedArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

@SuppressWarnings("Duplicates")
public class Main2 {
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
                .filter(new Column("_source.full_stack_trace").isNotNull())
                .withColumn("full_stack_trace_split", split(new Column("full_stack_trace"), "\n"));
        cleanText.show(10);


        StringIndexer indexExceptionClass = new StringIndexer().setInputCol("exception_class").setOutputCol("index_exception_class").setHandleInvalid("keep");
        StringIndexer indexErrorType = new StringIndexer().setInputCol("error_type").setOutputCol("index_error_type").setHandleInvalid("keep");
        StringIndexer indexErrorCode = new StringIndexer().setInputCol("error_code").setOutputCol("index_error_code").setHandleInvalid("keep");

        CountVectorizer cv = new CountVectorizer()
                .setInputCol("full_stack_trace_split")
                .setOutputCol("index_full_stack_trace")
                .setVocabSize(1000);

        IDF idf = new IDF().setInputCol("index_full_stack_trace")
                .setOutputCol("idf_index_full_stack_trace");

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(Arrays.asList("index_exception_class", "index_error_type", "index_error_code", "idf_index_full_stack_trace").toArray(new String[0]))
                .setOutputCol("features");

        PipelineStage[] pipelineStages = {indexErrorCode, indexErrorType, indexExceptionClass, cv, idf, vectorAssembler};
        Pipeline pipeline = new Pipeline().setStages(pipelineStages);
        PipelineModel model = pipeline.fit(cleanText);
        Dataset<Row> ds = model.transform(cleanText);
        ds.show();

        String[] vocabulary = new String[0];
        for (Transformer stage : model.stages()) {
            if (stage instanceof CountVectorizerModel){
                vocabulary = ((CountVectorizerModel) stage).vocabulary();
            }
            if (stage instanceof StringIndexerModel){
                String[] labels = ((StringIndexerModel) stage).labels();
            }
        }
        Broadcast<String[]> vocab_broadcast = sc.broadcast(vocabulary);


        LDA lda = new LDA().setK(25)
                .setSeed(321)
                .setOptimizer("em")
                .setFeaturesCol("features");

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

        datopicsMapped.show();


    }
}
