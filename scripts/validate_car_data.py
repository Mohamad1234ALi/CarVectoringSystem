import boto3
import pandas as pd
import sys
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.data_context.data_context.ephemeral_data_context import EphemeralDataContext

# -------------------------------
# CONFIG
# -------------------------------
DYNAMODB_TABLE = "CarList"
region_name = "us-east-1"

# -------------------------------
# Read all items from DynamoDB
# -------------------------------
boto3.setup_default_session(region_name=region_name)
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
# Great Expectations validation
# -------------------------------
context = EphemeralDataContext()  # ✅ create temporary context
engine = PandasExecutionEngine()
batch = Batch(data=df)
validator = Validator(execution_engine=engine, batches=[batch], data_context=context)

# Expectations
if "CarID" in df.columns:
    validator.expect_column_values_to_not_be_null("CarID")
    validator.expect_column_values_to_be_unique("CarID")

if "Mileage" in df.columns:
    validator.expect_column_values_to_be_between("Mileage", min_value=0, max_value=300000)

# Run validations
results = validator.validate()
success = results["success"]

print("\nValidation Results:")
print(success)

# Fail CI/CD if expectations are not met
if not success:
    sys.exit(1)
else:
    print("✅ Data validation passed successfully.")
