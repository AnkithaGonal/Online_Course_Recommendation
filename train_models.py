import pandas as pd
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from surprise import Dataset, Reader, SVD

# Load dataset

file_path = "online_course_recommendation_v2.xlsx"
df = pd.read_excel(file_path)

# Encode categorical columns
df_encoded = df.copy()
cat_cols = ['course_name','instructor','certification_offered',
            'difficulty_level','study_material_available']

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Features & Target
X = df_encoded.drop(['rating'], axis=1)
y = df_encoded['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Train Models
# -------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train_scaled, y_train)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id','course_id','rating']], reader)
trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)

# Save Models (in same env)

joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)

print("Models retrained and saved successfully.")
