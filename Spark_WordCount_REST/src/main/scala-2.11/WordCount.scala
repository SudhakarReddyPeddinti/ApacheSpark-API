import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by sudhakar on 2/10/17.
  */
object WordCount {
  def count(input_string:String): Array[(String, Int)] = {
    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("WordCount").setMaster("local[*]"))
    val input = sc.parallelize(List(input_string))
    val wc=input.flatMap(line=>{line.split(" ")}).map(word=>(word,1)).reduceByKey(_+_)
    val res=wc.collect()
    res
  }
}

