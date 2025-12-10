"""
get_data.py
-----------------
Downloads the Google Play Store dataset from Kaggle using kagglehub
and saves it into the local /data folder.

Run:
    pip install kagglehub[pandas-datasets]
    python get_data.py
"""

import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# File inside the dataset (update if different)
FILE_PATH = "googleplaystore.csv"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, FILE_PATH)


def main():
    # Create /data folder if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📥 Downloading dataset from Kaggle...")

    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "gauthamp10/google-playstore-apps",
            FILE_PATH,
        )
    except Exception as e:
        print("❌ Error downloading dataset:", e)
        return

    # Save locally
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Dataset downloaded and saved to {OUTPUT_FILE}")
    print("📊 Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
