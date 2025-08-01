import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Cleaned CSV ===
df = pd.read_csv(r"D:\elc_2nd\driver_cleaned.csv")
print("Total samples:", df.shape)

# === Split features and labels ===
X = df.drop('label', axis=1)
y = df['label']

# === Train-test split (stratified) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === Initialize LightGBM classifier ===
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

# === Train the model ===
print("🔄 Training the LightGBM model...")
model.fit(X_train, y_train)
print("✅ Training complete!")

# === Evaluate on test set ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === Save the trained model ===
joblib.dump(model, r"D:\elc_2nd\drowsiness_lgbm_model.pkl")
print("📦 Model saved as 'drowsiness_lgbm_model.pkl'")
