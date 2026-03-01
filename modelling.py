import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Setup MLflow Tracking (Lokal)
# Bagian ini tetap dikomentari agar tidak error di GitHub Actions (Kriteria 3)
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Dry_Bean_Classification")

# 2. Load Data Dry Bean
# Pastikan path file CSV sesuai dengan lokasi file di folder Anda
df = pd.read_csv('dry_bean_preprocessing/dry_bean_cleaned.csv') 

# Sesuaikan drop column dan target berdasarkan struktur dataset baru Anda
X = df.drop('Class', axis=1) # Target di dataset Anda bernama 'Class'
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Aktifkan Autolog
mlflow.sklearn.autolog()

# 4. Training Model
with mlflow.start_run():
    # Menggunakan RandomForestClassifier (Bagus untuk klasifikasi non-biner)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"Model Training Dry Bean Selesai. Accuracy: {acc}")

    mlflow.log_metric("accuracy", acc)
