import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_diabetes_data(
    input_file="output_diabetes.csv", output_file="output_diabetes_clean.csv"
):
    """
    Clean the synthetic diabetes data by removing invalid rows and fixing data types
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info(f"Original dataset shape: {df.shape}")

    # Store original shape for comparison
    original_shape = df.shape

    # 1. Fix data type issues
    logger.info("Fixing data types...")

    # Convert age to numeric, remove rows where age is not a number
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["age"])

    # Remove rows where age is unrealistic (outside 0-120 range)
    df = df[(df["age"] >= 0) & (df["age"] <= 120)]

    # Convert BMI to numeric, remove rows where BMI is not a number
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df = df.dropna(subset=["bmi"])

    # Remove rows where BMI is unrealistic (outside 10-100 range)
    df = df[(df["bmi"] >= 10) & (df["bmi"] <= 100)]

    # Convert HbA1c to numeric
    df["HbA1c_level"] = pd.to_numeric(df["HbA1c_level"], errors="coerce")
    df = df.dropna(subset=["HbA1c_level"])

    # Remove rows where HbA1c is unrealistic (outside 3-20 range)
    df = df[(df["HbA1c_level"] >= 3) & (df["HbA1c_level"] <= 20)]

    # Convert blood glucose to numeric
    df["blood_glucose_level"] = pd.to_numeric(
        df["blood_glucose_level"], errors="coerce"
    )
    df = df.dropna(subset=["blood_glucose_level"])

    # Remove rows where blood glucose is unrealistic (outside 50-500 range)
    df = df[(df["blood_glucose_level"] >= 50) & (df["blood_glucose_level"] <= 500)]

    # 2. Fix categorical variables
    logger.info("Fixing categorical variables...")

    # Ensure gender is valid
    valid_genders = ["Male", "Female"]
    df = df[df["gender"].isin(valid_genders)]

    # Ensure smoking_history is valid
    valid_smoking = ["never", "current", "former", "ever", "not current", "No Info"]
    df = df[df["smoking_history"].isin(valid_smoking)]

    # Ensure binary variables are 0 or 1
    binary_columns = ["hypertension", "heart_disease", "diabetes"]
    for col in binary_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col])
        df[col] = df[col].astype(int)
        df = df[df[col].isin([0, 1])]

    # 3. Additional data quality checks
    logger.info("Performing additional quality checks...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with all missing values
    df = df.dropna(how="all")

    # 4. Log cleaning results
    final_shape = df.shape
    rows_removed = original_shape[0] - final_shape[0]
    columns_removed = original_shape[1] - final_shape[1]

    logger.info(f"Cleaning completed:")
    logger.info(f"  - Original shape: {original_shape}")
    logger.info(f"  - Final shape: {final_shape}")
    logger.info(f"  - Rows removed: {rows_removed}")
    logger.info(f"  - Columns removed: {columns_removed}")
    logger.info(f"  - Retention rate: {final_shape[0]/original_shape[0]*100:.2f}%")

    # 5. Show sample statistics
    logger.info("Sample statistics after cleaning:")
    logger.info(f"  - Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
    logger.info(f"  - BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    logger.info(
        f"  - HbA1c range: {df['HbA1c_level'].min():.1f} - {df['HbA1c_level'].max():.1f}"
    )
    logger.info(
        f"  - Blood glucose range: {df['blood_glucose_level'].min():.1f} - {df['blood_glucose_level'].max():.1f}"
    )

    # 6. Save cleaned data
    logger.info(f"Saving cleaned data to {output_file}")
    df.to_csv(output_file, index=False)

    return df


def validate_data_quality(df, original_df=None):
    """
    Validate the quality of the cleaned data
    """
    logger.info("Validating data quality...")

    # Check for missing values
    missing_values = df.isnull().sum()
    logger.info("Missing values per column:")
    for col, missing in missing_values.items():
        if missing > 0:
            logger.info(f"  {col}: {missing}")

    # Check data types
    logger.info("Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")

    # Check value ranges
    logger.info("Value ranges for numerical columns:")
    numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    for col in numerical_cols:
        if col in df.columns:
            logger.info(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}")

    # Check categorical distributions
    logger.info("Categorical distributions:")
    categorical_cols = ["gender", "smoking_history"]
    for col in categorical_cols:
        if col in df.columns:
            logger.info(f"  {col}:")
            value_counts = df[col].value_counts()
            for value, count in value_counts.items():
                logger.info(f"    {value}: {count} ({count/len(df)*100:.1f}%)")

    # Compare with original if provided
    if original_df is not None:
        logger.info("Comparison with original data:")
        logger.info(f"  Original shape: {original_df.shape}")
        logger.info(f"  Cleaned shape: {df.shape}")
        logger.info(f"  Retention rate: {len(df)/len(original_df)*100:.2f}%")


if __name__ == "__main__":
    try:
        # Clean the data
        cleaned_df = clean_diabetes_data()

        # Validate the cleaned data
        validate_data_quality(cleaned_df)

        logger.info("Data cleaning completed successfully!")

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise e
