import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import json
from dotenv import load_dotenv


load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


mlflow.set_tracking_uri("https://dagshub.com/restuzaki/Eksperimen_SML_Damai.mlflow")

def main():
    print("Memuat dataset...")
    
    df = pd.read_csv('namadataset_preprocessing/dataset_clean.csv')
    
    
    df = df.sample(5000, random_state=42)

    X = df['URL']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Phishing_Detection_Tuning")

    with mlflow.start_run():
        print("Memulai Hyperparameter Tuning...")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        param_grid = {
            'tfidf__max_features': [500, 1000],
            'clf__C': [0.1, 1.0]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Tuning selesai. Akurasi Terbaik: {acc:.4f}")

       
        print("Menyimpan log dan artefak ke DagsHub...")
        
        mlflow.log_param("best_max_features", grid_search.best_params_['tfidf__max_features'])
        mlflow.log_param("best_C", grid_search.best_params_['clf__C'])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        
        mlflow.sklearn.log_model(best_model, "model")

        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix Phishing Detection")
        plt.ylabel('Aktual')
        plt.xlabel('Prediksi')
        plt.tight_layout()
        cm_path = "Membangun_model/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        
        params_path = "Membangun_model/best_params.json"
        with open(params_path, 'w') as f:
            json.dump(grid_search.best_params_, f)
        mlflow.log_artifact(params_path)

        print("Selesai!")

if __name__ == "__main__":
    main()