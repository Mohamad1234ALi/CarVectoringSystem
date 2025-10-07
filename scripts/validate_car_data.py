import boto3
import pandas as pd
import sys
from io import StringIO
import great_expectations as gx
from great_expectations.core.batch import Batch
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine import PandasExecutionEngine

# -------------------------------
# CONFIG: DynamoDB table and S3 paths
# -------------------------------
DYNAMODB_TABLE = "CarList"  # replace with your table name
# -------------------------------
# Read all items from DynamoDB
# -------------------------------
region_name = 'us-east-1'  # Your AWS region
boto3.setup_default_session(region_name=region_name)
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_TABLE)

print("Fetching data from DynamoDB...")
response = table.scan()
items = response['Items']

# Handle pagination if necessary
while 'LastEvaluatedKey' in response:
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
    items.extend(response['Items'])

if len(items) == 0:
    print("❌ No data found in DynamoDB!")
    sys.exit(1)

# Convert to DataFrame
df = pd.DataFrame(items)

# Ensure numeric columns are correct type
numeric_cols = ["Mileage", "Power", "FirstRegistration"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# -------------------------------
# Great Expectations validation
# -------------------------------
engine = PandasExecutionEngine()
batch = Batch(data=df)
validator = Validator(execution_engine=engine, batches=[batch])

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