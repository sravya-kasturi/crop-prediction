import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "crop.csv"
MODEL_PATH = BASE_DIR / "crop_model.pkl"

# 1. Load data
# Use the existing dataset in the same folder as this script
# so the program works regardless of the terminal's current working directory.
df = pd.read_csv(DATA_PATH)

# 2. Features and Target
# Note: Ensure column names match your CSV (N, P, K, temperature, humidity, ph, rainfall)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Save the model
with MODEL_PATH.open('wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved as {MODEL_PATH.name}!")