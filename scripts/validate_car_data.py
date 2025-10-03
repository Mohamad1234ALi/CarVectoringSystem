import boto3
import pandas as pd
import sys
from io import StringIO
import great_expectations as gx

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
# 1. Get a Data Context
# This is the entry point to your GX project
context = gx.get_context()

# 2. Add a Datasource for in-memory pandas DataFrames
datasource = context.sources.add_pandas(name="my_pandas_datasource")

# ... (Continuing from gx_example.py)

# 3. Create a Data Asset and get a Validator
# Use the RuntimeDataConnector for in-memory data
data_asset = datasource.add_dataframe_asset(
    name="my_dataframe_asset"
)

# Get a Validator for the DataFrame
validator = context.get_validator(
    batch_request=data_asset.build_batch_request(dataframe=df),
    create_expectation_suite_with_name="product_data_quality_suite"
)

# 4. Define expectations on your data
# Expectation 1: `CarID` column must be unique
validator.expect_column_values_to_not_be_null("CarID")
validator.expect_column_values_to_be_unique(column="CarID")
# Expectation 2: `Mileage` column must be between 0 and 300000
validator.expect_column_values_to_be_between(column="Mileage", min_value=0, max_value=300000)
# Expectation 3: `category` column values must be in a specific set
#validator.expect_column_values_to_be_in_set(column="category", value_set=["electronics", "apparel", "food"])

# 5. Save the Expectation Suite
# Saving the suite allows you to reuse it for future validations
validator.save_expectation_suite()


# ... (Continuing from gx_example.py)

# 6. Create and run a checkpoint to validate the data
checkpoint = gx.checkpoint.SimpleCheckpoint(
    name="my_simple_checkpoint",
    data_context=context,
    validator=validator,
)

checkpoint_result = checkpoint.run()

if not checkpoint_result["success"]:
    sys.exit(1)

# 7. Print the validation results
# The `success` flag shows if all expectations passed
print("\nValidation Result:")
print(f"Success: {checkpoint_result['success']}")

