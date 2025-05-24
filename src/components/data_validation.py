import pandas as pd

class DataValidator:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def check_missing_values(self):
        """Check for missing values in the dataset"""
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            print("Missing values detected in columns:")
            print(missing_values[missing_values > 0])
            return False
        else:
            print("No missing values detected.")
            return True

    def check_duplicates(self):
        """Check for duplicate rows in the dataset"""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Duplicate rows detected: {duplicates}")
            return False
        else:
            print("No duplicate rows detected.")
            return True
    
    def check_data_types(self):
        """Ensure the data types are as expected"""
        expected_data_types = {
            'column_name_1': 'float64',  # Replace with actual column names and expected types
            'column_name_2': 'int64'
        }
        for column, expected_type in expected_data_types.items():
            if column in self.df.columns and self.df[column].dtype != expected_type:
                print(f"Warning: Column '{column}' does not have expected type {expected_type}. Found: {self.df[column].dtype}")
                return False
        print("All columns have the expected data types.")
        return True

    def check_outliers(self, threshold=1.5):
        """Check for outliers using IQR (Interquartile Range)"""
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        
        # Outlier condition: values outside the range of Q1 - 1.5 * IQR and Q3 + 1.5 * IQR
        outliers = ((self.df < (Q1 - threshold * IQR)) | (self.df > (Q3 + threshold * IQR)))
        if outliers.any().any():
            print("Outliers detected:")
            print(outliers[outliers.any(axis=1)])
            return False
        else:
            print("No outliers detected.")
            return True

    def validate(self):
        """Run all validation checks"""
        if self.check_missing_values() and self.check_duplicates() and self.check_data_types() and self.check_outliers():
            print("Data validation passed successfully.")
            return True
        else:
            print("Data validation failed.")
            return False
