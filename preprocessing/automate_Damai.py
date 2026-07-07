import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import sys

def preprocess_data():
    print("Memulai proses otomatisasi preprocessing...")
    
   
    raw_path = 'namadataset_raw/phishing_site_urls.csv'
    output_dir = 'namadataset_preprocessing'
    output_path = os.path.join(output_dir, 'dataset_clean.csv')

    
    try:
        df = pd.read_csv(raw_path)
        print(f"Data mentah dimuat: {df.shape[0]} baris.")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {raw_path}")
        sys.exit(1)

    
    df_clean = df.drop_duplicates().dropna()
    print(f"Data setelah dibersihkan: {df_clean.shape[0]} baris.")

    
    encoder = LabelEncoder()
    df_clean['Label'] = encoder.fit_transform(df_clean['Label'])

    
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Data disimpan di: {output_path}")

if __name__ == "__main__":
    preprocess_data()