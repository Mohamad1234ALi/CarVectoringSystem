import boto3
import pandas as pd
import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.validator.validator import Validator
import sys
from io import StringIO

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
# Convert to Great Expectations DataFrame
# -------------------------------
context = ge.get_context()

# Create a Validator directly from Pandas DataFrame
validator = context.sources.pandas_default.read_batch(pd.DataFrame(df))

# -------------------------------
# Define expectations
# -------------------------------
validator.expect_column_values_to_not_be_null("CarID")
validator.expect_column_values_to_not_be_null("Mileage")
validator.expect_column_values_to_be_between("Mileage", 0, 300000)
validator.expect_column_values_to_not_be_null("FirstRegistration")
validator.expect_column_values_to_be_between("FirstRegistration", 1950, 2031)
validator.expect_column_values_to_not_be_null("Power")
validator.expect_column_values_to_be_between("Power", 0, 5000)

# -------------------------------
# Validate data
# -------------------------------
results = validator.validate()

if not results["success"]:
    print("❌ Data validation failed!")
    for r in results["results"]:
        if not r["success"]:
            print(f"Failed expectation: {r['expectation_config']['expectation_type']} on column {r['expectation_config']['kwargs'].get('column')}")
    sys.exit(1)
else:
    print("✅ Data validation passed!")

# -------------------------------
# Save validated data to trusted S3
# -------------------------------
# csv_buffer = StringIO()
# df.to_csv(csv_buffer, index=False)

# s3 = boto3.client('s3')
# s3.put_object(Bucket=TRUSTED_BUCKET, Key=TRUSTED_KEY, Body=csv_buffer.getvalue())
print("Validated data ")
