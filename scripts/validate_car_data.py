import boto3
import pandas as pd
import sys
import great_expectations as gx

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


def main():

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

    # Create an in-memory Great Expectations context
    context = gx.get_context(mode="ephemeral")

    # Convert DataFrame into a GE Dataset
    batch = gx.read_pandas(df)

    print("🔍 Running data quality checks...")

    # Define validation rules (expectations)
    batch.expect_column_values_to_not_be_null("Power")
    batch.expect_column_values_to_not_be_null("CarID")

    print("🧾 Validating dataset...")
    results = batch.validate()

    # Check for failures
    if not results["success"]:
        print("❌ Data validation failed!")
        for r in results["results"]:
            if not r["success"]:
                print(f" - {r['expectation_config']['expectation_type']}")
        sys.exit(1)
    else:
        print("✅ All data quality checks passed!")

if __name__ == "__main__":
    main()
