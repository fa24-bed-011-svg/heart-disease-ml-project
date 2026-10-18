
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("Heart Disease Prediction System")
st.write("Enter four attributes to get a model prediction.")
st.warning("Educational project only. Not a medical diagnosis.")

DATA_PATH = Path(__file__).parent / "processed.cleveland.data"

@st.cache_resource
def train_model():
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(
        DATA_PATH, header=None, names=columns, na_values="?"
    )
    df = df.dropna(subset=["age", "sex", "cp", "thalach", "target"])

    X = df[["age", "sex", "cp", "thalach"]].astype(float)
    y = (df["target"].astype(int) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

if not DATA_PATH.exists():
    st.error(
        "Dataset missing. Upload processed.cleveland.data "
        "to the same GitHub repository folder."
    )
    st.stop()

try:
    model, accuracy = train_model()
    st.caption(f"Test accuracy: {accuracy:.2%}")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox(
        "Sex", [0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )
    cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
    thalach = st.number_input("Maximum Heart Rate (thalach)", 50, 250, 150)

    if st.button("Predict Heart Disease", type="primary"):
        patient = pd.DataFrame(
            [[age, sex, cp, thalach]],
            columns=["age", "sex", "cp", "thalach"]
        )

        prediction = int(model.predict(patient)[0])

        if prediction == 1:
            st.error("Model prediction: Heart disease indicated")
        else:
            st.success("Model prediction: No heart disease indicated")

        st.caption("This result is not a medical diagnosis.")

except Exception as e:
    st.error("The app encountered an error:")
    st.exception(e)
