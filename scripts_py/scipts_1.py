# EDA_Play_store_Data_Logic.py
#
# Clean Python script containing all data cleaning, feature engineering, and 
# final correlation matrix setup logic from the Google Play Store EDA notebook.

# ==============================================================================
# 1. Importing Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Suppress warnings during cleaning

# ==============================================================================
# 2. Data Loading
# Note: Ensure this path is correct for your execution environment.
# ==============================================================================
try:
    df = pd.read_csv("E:\\HP user downloads\\Google-Playstore.csv.zip")
except FileNotFoundError:
    print("Error: The CSV file was not found. Please update the file path.")
    exit()

# ==============================================================================
# 3. Data Cleaning and Preprocessing
# ==============================================================================

# 3.1 Standardize Column Names (snake_case)
df.rename(lambda x: x.lower().strip().replace(' ', '_'),
            axis='columns', inplace=True)

# 3.2 Handling Missing Values (NaNs)

# Drop rows where essential core features are missing (low count NaNs)
df.dropna(subset=['installs','minimum_installs','currency','developer_email', 'editors_choice', 'in_app_purchases',
                  'ad_supported', 'content_rating','last_updated','scraped_time','developer_id','size','price',
                  'free','maximum_installs','category','app_id','app_name'], inplace=True)

# Drop columns with too many missing values (high count NaNs)
remove=['developer_website','privacy_policy']
df.drop(columns=remove, axis=1, inplace=True)

# Impute 'released' date with 'last_updated' date
df['released'] = pd.to_datetime(df['released'], errors='coerce')
df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
df['released'] = df['released'].fillna(df['last_updated'])

# Impute 'rating' and 'rating_count' with the median
filling_rating=df['rating'].median()
df['rating']=df['rating'].fillna(filling_rating)

filling_rating_count=df['rating_count'].median()
df['rating_count']=df['rating_count'].fillna(filling_rating_count)

# Drop rows with remaining missing values in 'minimum_android'
df.dropna(subset=['minimum_android'],inplace=True)

# 3.3 Data Type Conversion and Feature Engineering: 'size'

def con_size(size):
    """Converts app size string (with 'k' or 'M') to size in bytes."""
    if isinstance(size,str):
        size=size.replace(',','')
        if 'k' in size :
            return float(size.replace('k',''))*1024
        elif 'M' in size :
            return float(size.replace('M',''))*1024*1024
        elif 'Varies with device' in size:
            return np.nan
    return size

df['size']=df['size'].apply(con_size)
df.rename(columns={'size':'size_in_bytes'},inplace=True)

df['size_in_bytes']=pd.to_numeric(df['size_in_bytes'],errors='coerce')
df['size_in_Mb']=df['size_in_bytes']/(1024 * 1024)

# 3.4 Content Rating Standardization
df['content_rating'] = df['content_rating'].replace('Unrated',"Everyone")
df['content_rating'] = df['content_rating'].replace('Mature 17+',"Adults")
df['content_rating'] = df['content_rating'].replace('Adults only 18+',"Adults")
df['content_rating'] = df['content_rating'].replace('Everyone 10+',"Everyone")

# 3.5 Data Type Conversion: 'installs'
df['installs'] = df['installs'].astype(str).str.replace(',', '', regex=False)
df['installs'] = df['installs'].str.replace('+', '', regex=False)
df['installs'] = df['installs'].str.replace('Free', '0', regex=False)
df['installs'] = pd.to_numeric(df['installs'])

# 3.6 Feature Engineering: 'installs_category'
bins = [-1, 0, 10, 1000, 10000, 100000, 1000000, 10000000, 10000000000]
labels=['no', 'Very low', 'Low', 'Moderate', 'More than moderate', 'High', 'Very High', 'Top Notch']
df['installs_category'] = pd.cut(df['installs'], bins=bins, labels=labels)

# 3.7 Feature Engineering: 'type' (Free vs Paid)
# Correction check: Apps with price=0 must have Free=True
df.loc[(df.price==0) & (df.free==False),'free'] = True

# Create new 'type' column and drop 'free'
df['type']=df['free'].apply(lambda x:'free' if x else 'paid')
df.drop(columns=['free'],inplace=True)

# ==============================================================================
# 4. Correlation Matrix Preparation (Final Data Encoding)
# ==============================================================================

df_encoded = df.copy()

# Convert 'minimum_android' to numeric (e.g., '5.0 and up' -> 5.0)
df_encoded['minimum_android'] = df_encoded['minimum_android'].str.split(' ').str[0]
df_encoded['minimum_android'] = pd.to_numeric(df_encoded['minimum_android'], errors='coerce')

# Encode categorical columns using factorize
categorical_cols = ['category', 'installs_category', 'content_rating', 'currency', 'type']
for col in categorical_cols:
    df_encoded[col] = pd.factorize(df_encoded[col])[0]

# Select final numeric/encoded columns for correlation
numeric_cols = [
    'category', 'rating', 'rating_count', 'installs',
    'minimum_installs', 'maximum_installs', 'price',
    'size_in_bytes', 'minimum_android', 'ad_supported',
    'in_app_purchases', 'editors_choice', 'size_in_Mb',
    'installs_category', 'type'
]

# Calculate the correlation matrix
correlation_matrix = df_encoded[numeric_cols].corr()

# ==============================================================================
# 5. Output (Final DataFrame Ready for Analysis)
# ==============================================================================
print(f"Data Cleaning and Feature Engineering Complete. Final Data Shape: {df.shape}")

# df is the cleaned DataFrame
# correlation_matrix holds the final feature correlations

