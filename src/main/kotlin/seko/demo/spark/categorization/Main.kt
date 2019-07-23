package seko.demo.spark.categorization

class Main

fun main() {
    System.setProperty("hadoop.home.dir", "C:\\Users\\seko0716\\Documents\\projects\\demos\\spark_categorizer_demo")
    CategorizeService().categorize()
}