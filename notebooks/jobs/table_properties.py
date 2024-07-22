spark.sql("""
ALTER TABLE
    dev.databricks.slv_exam_guides 
SET TBLPROPERTIES 
    (delta.enableChangeDataFeed = true)
""")

spark.sql("""
ALTER TABLE
    dev.databricks.slv_documentation 
SET TBLPROPERTIES 
    (delta.enableChangeDataFeed = true)
""")
