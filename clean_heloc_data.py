import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("heloc_data_cleaning.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def clean_heloc_data(input_file="output_heloc.csv", output_file="output_heloc_clean.csv"):
    """
    Clean the HELOC synthetic data by removing invalid values and ensuring data quality
    """
    logger.info(f"Loading synthetic data from {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Initial data shape: {df.shape}")
        
        # Display initial data info
        logger.info("Initial data info:")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        logger.info(f"Missing values per column: {missing_values.to_dict()}")
        
        # Clean the data
        initial_rows = len(df)
        
        # 1. Handle RiskPerformance column (target variable)
        if "RiskPerformance" in df.columns:
            # Convert to proper format
            df["RiskPerformance"] = df["RiskPerformance"].astype(str)
            # Keep only valid values
            valid_risk_values = ["Good", "Bad", "good", "bad", "1", "0"]
            df = df[df["RiskPerformance"].isin(valid_risk_values)]
            
            # Standardize to Good/Bad
            df["RiskPerformance"] = df["RiskPerformance"].map({
                "Good": "Good", "good": "Good", "1": "Good",
                "Bad": "Bad", "bad": "Bad", "0": "Bad"
            })
        
        # 2. Clean numerical columns
        numerical_columns = []
        for col in df.columns:
            if col != "RiskPerformance":
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numerical_columns.append(col)
                except:
                    logger.warning(f"Could not convert column {col} to numeric")
        
        # 3. Handle special values in HELOC dataset
        # HELOC uses -7, -8, -9 as special values, keep them
        for col in numerical_columns:
            # Replace any non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Remove rows with too many missing values
        # Remove rows where more than 50% of numerical columns are missing
        threshold = len(numerical_columns) * 0.5
        df = df.dropna(thresh=len(df.columns) - threshold)
        
        # 5. Fill remaining missing values with median for numerical columns
        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median {median_val}")
        
        # 6. Remove rows with invalid RiskPerformance
        df = df.dropna(subset=["RiskPerformance"])
        
        # 7. Ensure all numerical values are within reasonable bounds
        # HELOC features typically have specific ranges
        for col in numerical_columns:
            # Remove extreme outliers (beyond 3 standard deviations)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Keep special values (-7, -8, -9) and values within bounds
            df = df[
                (df[col] >= lower_bound) | 
                (df[col] <= upper_bound) | 
                (df[col].isin([-7, -8, -9]))
            ]
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        logger.info(f"Data cleaning completed:")
        logger.info(f"Initial rows: {initial_rows}")
        logger.info(f"Final rows: {final_rows}")
        logger.info(f"Removed rows: {removed_rows} ({removed_rows/initial_rows*100:.2f}%)")
        
        # Check class distribution
        if "RiskPerformance" in df.columns:
            class_dist = df["RiskPerformance"].value_counts()
            logger.info(f"Class distribution after cleaning:")
            logger.info(f"  Good: {class_dist.get('Good', 0)} ({class_dist.get('Good', 0)/len(df)*100:.1f}%)")
            logger.info(f"  Bad: {class_dist.get('Bad', 0)} ({class_dist.get('Bad', 0)/len(df)*100:.1f}%)")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved to {output_file}")
        
        # Display final data info
        logger.info("Final data info:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise e


def validate_heloc_data(df):
    """
    Validate the cleaned HELOC data
    """
    logger.info("Validating cleaned HELOC data...")
    
    # Check data types
    logger.info("Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values found: {missing_values.to_dict()}")
    else:
        logger.info("No missing values found")
    
    # Check class distribution
    if "RiskPerformance" in df.columns:
        class_dist = df["RiskPerformance"].value_counts()
        logger.info(f"Class distribution: {class_dist.to_dict()}")
    
    # Check numerical column statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    logger.info("Numerical column statistics:")
    for col in numerical_cols:
        if col != "RiskPerformance":
            stats = df[col].describe()
            logger.info(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")


if __name__ == "__main__":
    try:
        logger.info("Starting HELOC data cleaning process...")
        
        # Clean the synthetic data
        cleaned_df = clean_heloc_data()
        
        # Validate the cleaned data
        validate_heloc_data(cleaned_df)
        
        logger.info("HELOC data cleaning completed successfully!")
        
    except Exception as e:
        logger.error(f"HELOC data cleaning failed: {str(e)}")
        raise e
