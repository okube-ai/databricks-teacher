from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
import pytesseract
import time
import os
import json
import openai


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #

exam_type = "engineer-professional"
exam_number = "05"
dirpath = f"/Users/osoucy/Documents/okube/databricks-exam-questions/{exam_type}-exam-{exam_number}"

client = openai.OpenAI(
    organization="org-SaovQqDsnEobPkVB0C8Adt1A",
)


# --------------------------------------------------------------------------- #
# Functions                                                                   #
# --------------------------------------------------------------------------- #


def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert("L")
    # Resize image
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2)
    # Apply binary thresholding
    image = image.point(lambda x: 0 if x < 140 else 255, "1")
    # Apply median filter
    image = image.filter(ImageFilter.MedianFilter())
    # Invert image (optional, depends on the quality of the original image)
    image = ImageOps.invert(image)
    return image


# --------------------------------------------------------------------------- #
# Execution                                                                   #
# --------------------------------------------------------------------------- #

filenames = list(os.listdir(dirpath))
filenames = [f for f in filenames if f.endswith(".png")]

n = len(filenames)
for i, filename in enumerate(filenames):

    filepath = os.path.join(dirpath, filename)
    output_filepath = filepath.replace(".png", ".json")

    if os.path.exists(output_filepath):
        continue

    print(f"Reading {i:04d}/{n:04d} {filename}... ", end="")

    with Image.open(filepath) as img:
        image = preprocess_image(img)

    text = pytesseract.image_to_string(image)

    print(f" Parsing... ", end="")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": """
             You are a text parser and you need to process text extracted from a image and return json string with
            this format:
            {"question": "...", "choices": {"A": "...", "B": "...", "C":"...", "D":"...", "E":"..."}}
            """,
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
    )

    try:
        d = json.loads(completion.choices[0].message.content)
    except json.decoder.JSONDecodeError:
        continue

    d["question_number"] = filename.replace(
        "q",
        "",
    ).replace(".png", "")
    d["exam_type"] = exam_type
    d["exam_number"] = exam_number

    if exam_type == "analyst-associate":

        if exam_number == "02" and d["question_number"] == "053":
            d["choices"]["D"] = "The amount of training data available"

        if (exam_number == "03" and d["question_number"] == "023") or (
            exam_number == "04" and d["question_number"] == "161"
        ):
            d["choices"]["D"] = d["choices"]["C"]
            d["choices"]["C"] = d["choices"]["B"]
            d["choices"]["B"] = d["choices"]["A"]
            d["choices"][
                "A"
            ] = "Use Scikit-learn for rapid prototyping and evaluate using R-squared."

        if (exam_number == "03" and d["question_number"] == "060") or (
            exam_number == "04" and d["question_number"] == "139"
        ):
            d["choices"] = {
                "A": "Databricks Visualizations with static widgets",
                "B": "Matplotlib embedded in HTML pages",
                "C": "Plotly Dash for creating interactive web applications",
                "D": "Seaborn plots within a Jupyter notebook",
            }

        if exam_number == "05" and d["question_number"] == "045":
            d["choices"] = {
                "A": "INNER JOIN",
                "B": "FULL JOIN",
                "C": "LEFT JOIN",
                "D": "ANTI JOIN",
                "E": "CROSS JOIN",
            }

        if exam_number == "05" and d["question_number"] == "025":
            d["choices"] = {
                "A": "A view is session-specific, while a temporary view persists across sessions.",
                "B": "Both views and temporary views store data physically in the workspace.",
                "C": "Temporary views allow data modifications, unlike regular views.",
                "D": "A view stores data physically, whereas a temporary view only exists during the session.",
                "E": "Both views and temporary views do not store data physically, but a view persists beyond the session.",
            }

        if exam_number == "05" and d["question_number"] == "037":
            d["choices"] = {
                "A": "By the type of data stored in the table.",
                "B": "By the size of the table data.",
                "C": "By the location of the data files specified in the table definition.",
                "D": "By the speed of query execution on the table.",
                "E": "By the number of columns in the table.",
            }

        if exam_number == "05" and d["question_number"] == "006":
            d["choices"] = {
                "A": "Historical data is maintained through periodic backups, accessed with the SHOW BACKUP command.",
                "B": "Delta Lake tables do not maintain historical data.",
                "C": "Historical data is stored in a dedicated audit log, accessed with the VIEW AUDIT LOG command.",
                "D": "Historical data is maintained through versioned table updates, accessed with the DESCRIBE HISTORY command.",
                "E": "Historical data is maintained in separate snapshot tables, accessed with the SHOW SNAPSHOT command.",
            }

        if exam_number == "05" and d["question_number"] == "003":
            d["choices"]["E"] = "SELECT Region, Product, SUM(SalesAmount) AS TotalSales FROM SalesData GROUP BY GROUPING SETS ((Region, Product), (Region), ());"

        if exam_number == "06" and d["question_number"] == "045":
            d["choices"] = {
                "A": "Delta Lake tables do not maintain historical data.",
                "B": "By creating a new table for each update.",
                "C": "By storing historical data in a separate cloud storage.",
                "D": "Through automatic backups at regular intervals.",
                "E": "By maintaining a versioned history of data changes for a configurable period of time.",
            }

        if exam_number == "06" and d["question_number"] == "042":
            d["choices"] = {
                "A": "Use of persistent tables instead of temporary views.",
                "B": "Increased hardware resources allocation.",
                "C": "Caching of intermediate data and results from previous query executions.",
                "D": "Improved data indexing mechanisms.",
                "E": "Automatic query rewriting for optimization.",
            }

        if exam_number == "06" and d["question_number"] == "024":
            d["choices"] = {
                "A": "SELECT * FROM Employees RIGHT JOIN Departments ON Employees.EmployeeID = Departments.EmployeeID;",
                "B": "SELECT * FROM Employees LEFT JOIN Departments ON Employees. EmployeeID = Departments. EmployeeID;",
                "C": "SELECT Employees.Name, Departments.Department FROM Employees LEFT JOIN Departments ON Employees.EmployeeID = Departments.EmployeeID;",
                "D": "SELECT Employees.Name, Departments.Department FROM Employees FULL JOIN Departments ON Employees.EmployeeID = Departments.EmployeeID;",
                "E": "SELECT Employees.Name, Departments.Department FROM Employees RIGHT JOIN Departments ON Employees.EmployeeID = Departments.EmployeeID;",
            }
        if exam_number == "06" and d["question_number"] == "022":
            continue

        if exam_number == "06" and d["question_number"] == "008":
            d["choices"] = {
                "A": "Modify Permissions.",
                "B": "Owner.",
                "C": "Can Edit.",
                "D": "Can View.",
                "E": "No permissions",
            }

    if exam_type == "engineer-associate":

        if exam_number == "01" and d["question_number"] == "042":
            d["choices"] = {
                "A": "Data Studio",
                "B": "Cluster event log",
                "C": "Workflows",
                "D": "DBFS",
                "E": "Data Explorer",
            }

        if exam_number == "01" and d["question_number"] == "030":
            d["choices"] = {
                "A": "spark.readStream('events')",
                "B": "spark.read.table('events')",
                "C": "spark.readStream.table('events')",
                "D": "spark.readStream().table('events')",
                "E": "spark.stream.read('events)",
            }

        if exam_number == "01" and d["question_number"] == "038":
            d["choices"] = {
                "A": "Slack",
                "B": "Webhook",
                "C": "SMS",
                "D": "Microsoft Teams",
                "E": "Email",
            }

        if exam_number == "01" and d["question_number"] == "037":
            d["choices"] = {
                "A": "Slack",
                "B": "Webhook",
                "C": "SMS",
                "D": "Microsoft Teams",
                "E": "Email",
            }

        if exam_number == "01" and d["question_number"] == "005":
            d["choices"] = {
                "A": "Databricks web application",
                "B": "Notebooks",
                "C": "Repos",
                "D": "Cluster virtual machines",
                "E": "Workflows",
            }

        if exam_number == "01" and d["question_number"] == "004":
            d["choices"] = {
                "A": "Clone",
                "B": "Commit",
                "C": "Merge",
                "D": "Push",
                "E": "Pull",
            }
        if exam_number == "01" and d["question_number"] == "013":
            d["choices"] = {
                "A": "if process_mode = 'init' & not is_table_exist: print('Start processing...')",
                "B": "if process_mode = 'init' and not is_table_exist = True: print('Start processing...')",
                "C": "if process_mode = 'init' and is_table_exist = False: print('Start processing...')",
                "D": "if (process_mode = 'init') and (not is_table_exist): print('Start processing...')",
                "E": "if process_mode == 'init' and not is_table_exist: print('Start processing...')",
            }

        if exam_number == "01" and d["question_number"] == "002":
            d["choices"] = {
                "A": "Delta",
                "B": "Parquet",
                "C": "JSON",
                "D": "Hive-specific format",
                "E": "Both, Parquet and JSON",
            }

        if exam_number == "02" and d["question_number"] == "010":
            d["choices"] = {
                "A": "The deleted data files were larger than the default size threshold. While the remaining files are smaller than the default size threshold and can not be deleted.",
                "B": "The deleted data files were smaller than the default size threshold. While the remaining files are larger than the default size threshold and can not be deleted.",
                "C": "The deleted data files were older than the default retention threshold. While the remaining files are newer than the default retention threshold and can not be deleted.",
                "D": "The deleted data files were newer than the default retention threshold. While the remaining files are older than the default retention threshold and can not be deleted.",
                "E": "More information is needed to determine the correct answer",
            }

        if exam_number == "02" and d["question_number"] == "004":
            d["choices"] = {
                "A": "Delta",
                "B": "Parquet",
                "C": "JSON",
                "D": "Hive-specific format",
                "E": "XML",
            }

    if exam_type == "engineer-professional":

        if exam_number == "01" and (d["question_number"] == "031" or d["question_number"] == "032"):
            d["choices"] = {
                "A": "Newly updated records will be merged into the target table, modifying previous entries with the same primary keys.",
                "B": "Newly updated records will be appended to the target table.",
                "C": "Newly updated records will overwrite the target table.",
                "D": "The entire history of updated records will be appended to the target table at each execution, which leads to duplicate entries.",
                "E": "The entire history of updated records will overwrite the target table at each execution.",
            }

        if exam_number == "01" and d["question_number"] == "022":
            d["choices"] = {
                "A": "Static Delta tables must be small enough to be broadcasted to all worker nodes in the cluster.",
                "B": "Static Delta tables need to be partitioned in order to be used in stream-static join.",
                "C": "Static Delta tables need to be refreshed with REFRESH TABLE command for each microbatch of a stream-static join",
                "D": "The latest version of the static Delta table is returned each time it is queried by a microbatch of the stream-static join",
                "E": "The latest version of the static Delta table is returned only for the first microbatch of the stream-static join. Then, it will be cached to be used by any upcoming microbatch.",
            }

        if exam_number == "01" and d["question_number"] == "001":
            d["choices"] = {
                "A": "dbutils.secrets",
                "B": "dbutils.library",
                "C": "dbutils.fs",
                "D": "dbutils.notebook",
                "E": "dbutils.widgets",
            }

        if exam_number == "01" and d["question_number"] == "059":
            d["choices"] = {
                "A": "job_id",
                "B": "run_id",
                "C": "run_key",
                "D": "task_id",
                "E": "task_key",
            }

        if exam_number == "01" and d["question_number"] == "060":
            d["choices"] = {
                "A": "Task 1 will succeed. Task 2 will partially fail. Task 3 will be skipped",
                "B": "Task 1 will succeed. Task 2 will completely fail. Task 3 will be skipped",
                "C": "Tasks 1 and 3 will succeed, while Task 2 will partially fail",
                "D": "Tasks 1 and 3 will succeed, while Task 2 will completely fail",
                "E": "All tasks will completely fail",
            }

        if exam_number == "01" and d["question_number"] == "048":
            d["choices"] = {
                "A": "MANAGE permission on the 'DataOps-Prod' scope",
                "B": "READ permission on the 'DataOps-Prod' scope",
                "C": "MANAGE permission on each secret in the 'DataOps-Prod' scope",
                "D": "READ permission on each secret in the 'DataOps-Prod' scope",
                "E": "Workspace Administrator role",
            }

        if exam_number == "02" and d["question_number"] == "052":
            d["choices"] = {
                "A": "workspace",
                "B": "fs",
                "C": "jobs",
                "D": "configure",
                "E": "libraries",
            }

        if exam_number == "02" and d["question_number"] == "045":
            d["choices"] = {
                "A": "Query duration",
                "B": "Query execution time",
                "C": "Succeeded Jobs",
                "D": "Spill size",
                "E": "Number of input rows",
            }

        if exam_number == "02" and d["question_number"] == "005":
            d["choices"] = {
                "A": "Delta",
                "B": "Parquet",
                "C": "JSON",
                "D": "Hive-specific format",
                "E": "Both, Parquet and JSON",
            }

        if exam_number == "02" and d["question_number"] == "004":
            d["choices"] = {
                "A": "test_df.apply(predict_udf, *column_list).select('record_id', 'prediction')",
                "B": "test_df.select('record_id', predict_udf(*column_list).alias('prediction'))",
                "C": "predict_udf('record_id', test_df).select('record _id', 'prediction')",
                "D": "mlflow.pyfunc.map(predict_udf, test_df, 'record_id').alias('prediction')",
                "E": "mlflow.pyfunc.map(predict_udf, test_df, 'record_id').alias('prediction')",
            }

        if exam_number == "02" and d["question_number"] == "016":
            d["choices"] = {
                "A": "option('cloudFiles.schemaEvolutionMode', 'addNewColumns')",
                "B": "option('cloudFiles.mergeSchema', True)",
                "C": "option('mergeSchema', True)",
                "D": "schema(schema_definition, mergeSchema=True)",
                "E": "Autoloader can not automatically evolve the schema of the table when new fields are detected",
            }

        if exam_number == "02" and d["question_number"] == "015":
            d["choices"] = {
                "A": "Newly updated records will be merged into the target table, modifying previous entries with the same primary keys",
                "B": "Newly updated records will be appended to the target table.",
                "C": "Newly updated records will overwrite the target table.",
                "D": "The entire history of updated records will be appended to the target table at each execution, which leads to duplicate entries",
                "E": "The entire history of updated records will overwrite the target table at each execution.",
            }

        if exam_number == "02" and d["question_number"] == "029":
            d["choices"] = {
                "A": "key",
                "B": "value",
                "C": "topic",
                "D": "partition",
                "E": "timestamp",
            }

        if exam_number == "02" and d["question_number"] == "058":
            d["choices"] = {
                "A": "Task 1 will partially fail. Tasks 2 and 3 will be skipped",
                "B": "Task 1 will partially fail. Tasks 2 and 3 will run and succeed",
                "C": "Task 1 will completely fail. Tasks 2 and 3 will be skipped",
                "D": "Task 1 will completely fail. Tasks 2 and 3 will run and succeed",
                "E": "All tasks will partially fail",
            }

        if exam_number == "03" and d["question_number"] == "051":
            d["choices"] = {
                "A": "1, 2",
                "B": "1, 2, 3 ",
                "C": "2, 3",
                "D": "1",
                "E": "1, 3",
            }

        if exam_number == "03" and d["question_number"] == "024":
            d["choices"] = {
                "A": "foreachBatch()",
                "B": "foreachUDF()",
                "C": "pivot()",
                "D": "explain()",
                "E": "fun()",
            }

        if exam_number == "03" and d["question_number"] == "031":
            d["choices"] = {
                "A": "ALTER orders DROP PRIMARY KEY CASCADE;",
                "B": "ALTER TABLE orders DROP PRIMARY KEY CASCADE;",
                "C": "ALTER TABLE orders DROP PRIMARY_KEY;",
                "D": "ALTER orders DROP PRIMARY_KEY;",
                "E": "ALTER TABLE orders DROP PRIMARY_KEY CASCADE;",
            }

        if exam_number == "03" and d["question_number"] == "023":
            d["choices"] = {
                "A": "Only SQL",
                "B": "Scala and Python",
                "C": "Python, Java and Scala",
                "D": "Python, SQL and Scala",
                "E": "Python and SQL",
            }
        if exam_number == "03" and d["question_number"] == "048":
            d["choices"] = {
                "A": "Is Creator permission",
                "B": "Is Owner permission",
                "C": "Can Manage permission",
                "D": "Admin permission",
                "E": "No permission",
            }

        if exam_number == "04" and d["question_number"] == "053":
            d["choices"] = {
                "A": "Jobs",
                "B": "Stages",
                "C": "Structured Streaming",
                "D": "JDBC/ODBC Server",
                "E": "Storage",
            }

        if exam_number == "04" and d["question_number"] == "030":
            d["choices"] = {
                "A": "Catalog",
                "B": "Database",
                "C": "Table",
                "D": "View",
                "E": "Schema",
            }

        if exam_number == "04" and d["question_number"] == "025":
            d["choices"] = {
                "A": "_delta_log",
                "B": "_ratings_log",
                "C": "_transaction_log",
                "D": "_log",
                "E": "_log_ratings",
            }

        if exam_number == "04" and d["question_number"] == "022":
            d["choices"] = {
                "A": "dropNewColumns",
                "B": "addNewColumns",
                "C": "ignoreNewColumns",
                "D": "none",
                "E": "failOnNewColumns",
            }

        if exam_number == "04" and d["question_number"] == "009":
            d["choices"] = {
                "A": "{'name': 'new_job'}",
                "B": "{'existing_cluster_id': '8522-150723-mA1016', 'notebook_task':{'notebook_path': 'path/to/notebook'}}",
                "C": "{'name': 'new_job', 'notebook_task':{'notebook_path': 'path/to/notebook'}}",
                "D": "All three will return an error",
                "E": "All three will be successful",
            }

        if exam_number == "04" and d["question_number"] == "058":
            d["choices"] = {
                "A": "60 days",
                "B": "45 days",
                "C": "No version history is visible if the notebook is not attached to Git",
                "D": "The version history can be accessed from the time the notebook was created",
                "E": "30 days",
            }

        if exam_number == "05" and d["question_number"] == "047":
            d["choices"] = {
                "A": "passport_number",
                "B": "credit_card_number",
                "C": "biometrics",
                "D": "gender",
                "E": "name",
            }

        if exam_number == "05" and d["question_number"] == "052":
            d["choices"] = {
                "A": "right_outer join is not supported in Spark.",
                "B": "You cannot use multiple select operations in a single line of code.",
                "C": "All the columns from both the DataFrames cannot be selected.",
                "D": "medals column exists in both the DataFrames",
                "E": "medals column does not exist in any of the DataFrames being joined",
            }

        if exam_number == "05" and d["question_number"] == "009":
            d["choices"] = {
                "A": "databricks run cancel --run-id 2795",
                "B": "databricks runs cancel --job-id 2795 --run-id 96746",
                "C": "databricks run cancel —-job-name fetch_details --run-id 96746",
                "D": "databricks runs cancel --run-id 96746",
                "E": "databricks run cancel --job-id 2795 --run-id 96746",
            }

        if exam_number == "05" and d["question_number"] == "004":
            d["choices"] = {
                "A": "The job will be created successfully with both the jobs named new_job",
                "B": "The job will be created successfully with the second job named new_job_1",
                "C": "The job will not be created as a job with the same name already exists.",
                "D": "The job will be created successfully by overwriting the previous job as two jobs cannot share a name",
                "E": "The task in the second job will be appended to the existing job.",
            }

        if exam_number == "05" and d["question_number"] == "014":
            d["choices"] = {
                "A": "dropNewColumns",
                "B": "none",
                "C": "rescue",
                "D": "Any one from rescue or none can be used",
                "E": "Any one from rescue or dropNewColumns can be used",
            }

    try:
        d["question"]
        d["choices"]["A"]
        d["choices"]["B"]
        d["choices"]["C"]
        d["choices"]["D"]
        d["choices"]["E"]
    except KeyError as e:
        print(exam_type, exam_number, d["question_number"])
        print(text)
        print(d)
        raise e

    with open(filepath.replace(".png", ".json"), "w") as fp:
        json.dump(d, fp)

    print(f"done.")
    time.sleep(1.0)
