from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col

# Create Spark app
spark = SparkSession.builder \
    .appName("Simple Word Count Project") \
    .getOrCreate()

# Sample data
data = [
    ("Spark is fast and powerful",),
    ("Python with Spark is useful",),
    ("Spark can process big data",),
]

# Create DataFrame
df = spark.createDataFrame(data, ["sentence"])

# Word count
word_count = df.select(
    explode(split(lower(col("sentence")), " ")).alias("word")
).groupBy("word").count().orderBy("count", ascending=False)

# Show result
word_count.show()

# Stop Spark
spark.stop()
