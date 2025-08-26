import streamlit as st
import pandas as pd
import joblib
import pickle

# --------------------------
# Load Data & Models
# --------------------------
@st.cache_resource
def load_models():
    rf_model = joblib.load("random_forest_model.pkl")
    xgb_model = joblib.load("xgboost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")

    # Original data
    df = pd.read_excel("online_course_recommendation_v2.xlsx")

    # Encoded dataset
    df_encoded = pd.read_csv("df_encoded.csv")

    return df, df_encoded, rf_model, xgb_model, scaler, encoders


df, df_encoded, rf_model, xgb_model, scaler, encoders = load_models()

# --------------------------
# Recommendation Function
# --------------------------
def hybrid_recommend(user_id, top_n=5, difficulty=None, certification=None):
    if "user_id" not in df_encoded.columns:
        return pd.DataFrame({"Error": ["user_id column not found in df_encoded"]})

    if user_id not in df_encoded["user_id"].unique():
        return pd.DataFrame({"Error": ["User ID not found in dataset"]})

    # Courses already taken by user
    user_courses = df_encoded[df_encoded["user_id"] == user_id]["course_id"].values
    candidate_courses = df_encoded[~df_encoded["course_id"].isin(user_courses)].copy()

    if candidate_courses.empty:
        return pd.DataFrame({"Error": ["No new courses to recommend"]})

    # ✅ Use the same features as in training
    feature_names = scaler.feature_names_in_
    X = candidate_courses[feature_names]

    # Scale
    X_scaled = scaler.transform(X)

    # Predictions
    rf_preds = rf_model.predict(X_scaled)
    xgb_preds = xgb_model.predict(X_scaled)

    # Hybrid score
    candidate_courses["pred_score"] = 0.5 * rf_preds + 0.5 * xgb_preds

    # ✅ Merge with df to get human-readable fields (avoids inverse_transform length issues)
    recs = candidate_courses.merge(
        df[
            [
                "course_id",
                "course_name",
                "instructor",
                "difficulty_level",
                "certification_offered",
                "rating",
                "course_price"
            ]
        ],
        on="course_id",
        how="left"
    )

    # Apply filters safely
    if difficulty and "difficulty_level" in recs.columns:
        recs = recs[recs["difficulty_level"] == difficulty]
    if certification and "certification_offered" in recs.columns:
        recs = recs[recs["certification_offered"] == certification]

    recs = recs.sort_values("pred_score", ascending=False)

    return recs.head(top_n)


# --------------------------
# Streamlit UI
# --------------------------
st.title("🎓 Online Course Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1, step=1)

top_n = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

difficulty = st.selectbox(
    "Select Difficulty Level (optional):",
    options=[""] + sorted(df["difficulty_level"].dropna().unique().tolist())
)
difficulty = difficulty if difficulty else None

certification = st.selectbox(
    "Certification Offered? (optional):",
    options=["", "Yes", "No"]
)
certification = certification if certification else None

if st.button("Get Recommendations"):
    recs = hybrid_recommend(user_id, top_n, difficulty, certification)

    if "Error" in recs.columns:
        st.error(recs["Error"].iloc[0])
    else:
        st.subheader("✅ Recommended Courses:")

        # ✅ Show only columns that actually exist
        display_cols = [
            "course_id",
            "course_name",
            "instructor",
            "difficulty_level",
            "certification_offered",
            "pred_score",
            "rating",
            "course_price"
        ]
        available_cols = [col for col in display_cols if col in recs.columns]

        st.dataframe(recs[available_cols])

