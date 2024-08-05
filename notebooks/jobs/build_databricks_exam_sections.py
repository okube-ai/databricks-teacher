import pyspark.sql.functions as F
import pyspark.sql.types as T


# --------------------------------------------------------------------------- #
# Build Data                                                                  #
# --------------------------------------------------------------------------- #

# Define schema
schema = T.StructType(
    [
        T.StructField("exam_type", T.StringType(), True),
        T.StructField(
            "section",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("index", T.IntegerType(), True),
                        T.StructField("name", T.StringType(), True),
                        T.StructField("weight", T.DoubleType(), True),
                    ]
                ),
                True,
            ),
            True,
        ),
    ]
)


# Create data
data = [
    (
        "analyst-associate",
        [
            {"index": 1, "name": "Databricks SQL", "weight": 0.22},
            {"index": 2, "name": "Data Management", "weight": 0.2},
            {"index": 3, "name": "SQL in the Lakehouse", "weight": 0.29},
            {"index": 4, "name": "Data Visualization and Dashboarding", "weight": 0.18},
            {"index": 5, "name": "Analytics and Applications", "weight": 0.11},
        ],
    ),
    (
        "engineer-associate",
        [
            {"index": 1, "name": "Databricks Lakehouse Platform", "weight": 0.24},
            {"index": 2, "name": "ELT with Apache Spark", "weight": 0.29},
            {"index": 3, "name": "Incremental Data Processing", "weight": 0.22},
            {"index": 4, "name": "Production Pipelines", "weight": 0.16},
            {"index": 5, "name": "Data Governance", "weight": 0.09},
        ],
    ),
    (
        "engineer-professional",
        [
            {"index": 1, "name": "Databricks Tooling", "weight": 0.2},
            {
                "index": 2,
                "name": "Data Processing (Batch Processing, Incremental processing and Optimization)",
                "weight": 0.3,
            },
            {"index": 3, "name": "Data Modeling", "weight": 0.2},
            {"index": 4, "name": "Security & Governance", "weight": 0.1},
            {"index": 5, "name": "Monitoring & Logging", "weight": 0.1},
            {"index": 6, "name": "Testing & Deployment", "weight": 0.1},
        ],
    ),
    (
        "ml-associate",
        [
            {"index": 1, "name": "Databricks ML", "weight": 0.29},
            {"index": 2, "name": "ML Workflows", "weight": 0.29},
            {"index": 3, "name": "Spark ML", "weight": 0.33},
            {"index": 4, "name": "Scaling ML Models", "weight": 0.09},
        ],
    ),
    (
        "ml-professional",
        [
            {"index": 1, "name": "Experimentation", "weight": 0.3},
            {"index": 2, "name": "Model Lifecycle Management", "weight": 0.3},
            {"index": 3, "name": "Model Deployment", "weight": 0.25},
            {"index": 4, "name": "Solution and Data Monitoring", "weight": 0.15},
        ],
    ),
    (
        "genai-associate",
        [
            {"index": 1, "name": "Design Applications", "weight": 0.14},
            {"index": 2, "name": "Data Preparation", "weight": 0.14},
            {"index": 3, "name": "Application Development", "weight": 0.3},
            {
                "index": 4,
                "name": "Assembling and Deploying Applications",
                "weight": 0.22,
            },
            {"index": 5, "name": "Governance", "weight": 0.08},
            {"index": 6, "name": "Evaluating and Monitoring", "weight": 0.12},
        ],
    ),
]

# Create DataFrame
df = spark.createDataFrame(data, schema)
df = df.withColumn("section", F.explode("section"))
df = df.withColumn("index", F.col("section.index"))
df = df.withColumn("name", F.col("section.name"))
df = df.withColumn("weight", F.col("section.weight"))
df = df.drop("section")

display(df)

# --------------------------------------------------------------------------- #
# Export                                                                      #
# --------------------------------------------------------------------------- #

pdf = df.toPandas()
pdf.to_json(
    "/Volumes/dev/sources/landing/events/databricks/exam-sections/sections.json",
    orient="records",
)
