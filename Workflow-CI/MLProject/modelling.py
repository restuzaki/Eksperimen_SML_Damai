import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn


os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")


DAGSHUB_USERNAME = "restuzaki"
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/Eksperimen_SML_Damai.mlflow")

def main():
    print("Memuat dataset untuk CI Retraining...")
   
    df = pd.read_csv('namadataset_preprocessing/dataset_clean.csv')
    
    X = df['URL']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Phishing_Detection_CI_Pipeline_V2")

    with mlflow.start_run() as run:
        print("Melatih model dengan parameter terbaik...")
        
       
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500)),
            ('clf', LogisticRegression(C=1.0, max_iter=1000))
        ])

        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Retraining selesai. Akurasi: {acc:.4f}")

       
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log model ke MLflow
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"Model berhasil disimpan di Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()