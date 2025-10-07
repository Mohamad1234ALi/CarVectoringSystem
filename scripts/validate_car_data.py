import boto3
import pandas as pd
import sys
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.data_context.data_context.ephemeral_data_context import EphemeralDataContext
from great_expectations.data_context.types.base import DataContextConfig

# -------------------------------
# CONFIG
# -------------------------------
DYNAMODB_TABLE = "CarList"
REGION = "us-east-1"

# -------------------------------
# Read items from DynamoDB
# -------------------------------
boto3.setup_default_session(region_name=REGION)
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE)

print("Fetching data from DynamoDB...")
response = table.scan()
items = response["Items"]

while "LastEvaluatedKey" in response:
    response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
    items.extend(response["Items"])

if not items:
    print("❌ No data found in DynamoDB!")
    sys.exit(1)

df = pd.DataFrame(items)

# Convert numeric columns
numeric_cols = ["Mileage", "Power"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# Great Expectations Validation
# -------------------------------

# 1️⃣ Create minimal in-memory DataContext config
project_config = DataContextConfig(
    datasources={},
    store_backend_defaults=None,
    anonymous_usage_statistics={"enabled": False},
)

context = EphemeralDataContext(project_config=project_config)

# 2️⃣ Create execution engine + validator
engine = PandasExecutionEngine()
batch = Batch(data=df)
validator = Validator(execution_engine=engine, batches=[batch], data_context=context)

# 3️⃣ Define expectations
if "CarID" in df.columns:
    validator.expect_column_values_to_not_be_null("CarID")
    validator.expect_column_values_to_be_unique("CarID")

if "Mileage" in df.columns:
    validator.expect_column_values_to_be_between("Mileage", min_value=0, max_value=300000)

# 4️⃣ Run validation
results = validator.validate()
success = results["success"]

print("\n✅ Validation completed.")
print(f"Validation success: {success}")

# 5️⃣ Fail CI/CD if validation fails
if not success:
    print("❌ Data quality checks failed. Stopping deployment.")
    sys.exit(1)
else:
    print("✅ All data quality checks passed successfully.")
