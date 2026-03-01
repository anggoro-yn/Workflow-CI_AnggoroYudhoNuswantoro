import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data (Pastikan folder 'iris_preprocessing' ada di GitHub)
df = pd.read_csv('iris_preprocessing/iris_cleaned.csv')
X = df.drop('variety', axis=1)
y = df['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Aktifkan Autolog
mlflow.sklearn.autolog()

# 3. Training Model
# start_run() tanpa argumen akan otomatis menggunakan Run ID dari 'mlflow run'
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Model Training Selesai. Accuracy: {acc}")
