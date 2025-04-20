import streamlit as st
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,roc_curve
model=joblib.load("movie_success_model.pkl")
st.set_page_config(page_title="Movie Success Predictor", layout="wide")

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/bmp;base64,{encoded_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use the function with your uploaded image
set_background(r"C:\Users\nagab\Downloads\pexels-photo-8263325.webp")

st.title("🎬 Movie Success Prediction App")
st.markdown("**Predict whether a movie will be successful based on key features.**")

tab1, tab2, tab3= st.tabs(["📥 Database", "📊 Prediction Interface","📊 Graphs"])

with tab1:
    @st.cache_data
    def load_data():
        df = pd.read_csv(r"C:\Users\nagab\OneDrive\Desktop\ml_1\ml(movie)\tmbd_movies_df.csv")  # your final dataset
        return df

    df = load_data()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Train Your Model")

    features = st.multiselect("Select features to use:", df.columns.drop("is_sucssfull"),
                               default=["budget", "popularity", "runtime"])
    target = "is_sucssfull"

    if features:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_data = pd.concat([X_train, y_train], axis=1)

        train_data = train_data.dropna()

        # Split them back
        X_train = train_data.drop("is_sucssfull", axis=1)
        y_train = train_data["is_sucssfull"]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        st.success("Model trained successfully!")

        X_test_copy = X_test.copy()
        X_test_copy["target"] = y_test

        # Drop rows with missing values
        X_test_copy = X_test_copy.dropna()

        # Split them back again
        X_test_clean = X_test_copy.drop("target", axis=1)
        y_test_clean = X_test_copy["target"]


        y_pred = model.predict(X_test_clean)
        st.text("Classification Report:")
        st.text(classification_report(y_test_clean, y_pred))
    
    col1,col2=st.columns(2)
    with col1:
        if st.button("Best fit line-budget vs revnue"):
            movies_df=pd.read_csv(r"C:\Users\nagab\OneDrive\Desktop\ml_1\ml(movie)\tmbd_movies_df.csv")
            sle.regplot(x=movies_df['budget'],y=movies_df['revenue'])
            plt.title("Best fit line-budget vs revnue")
            plt.show()
            st.pyplot(plt)

with tab2:
    st.subheader("🎬 Movie Success Prediction")

    # ✅ Get user input
    budget = st.number_input("💰 Budget (in $)", value=0)
    runtime = st.number_input("⏱️ Runtime (min)", value=0) 
    popularity = st.number_input("🔥 Popularity", value=0) 
    vote_average=st.number_input('vote_average',value=0)
    vote_count=st.number_input('vote_count',value=0)
    release_year=st.number_input("release_year" ,value=0)

    
    if st.button("🚀 Predict Success"):
        input_data = {
            "budget": budget,
            "runtime": runtime,
            "popularity": popularity,
            "vote_average":vote_average,
            "vote_count":vote_count,
            "release_year":release_year    
        }

        input_df = pd.DataFrame([input_data])
        st.write("🔍 Input Preview:", input_df)

        try:
            features = ['budget','runtime',"popularity",'vote_average','vote_count',"release_year"]
            input_df = input_df[features].fillna(0)
            prediction = model.predict(input_df)[0]
            st.success("🎯 Prediction: " + ("✅ Successful" if prediction == 1 else "❌ Not Successful"))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab3:
    st.header("🎬 Movie Genre Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎯 Show Top 5 Genres with Most Movies"):
            st.subheader("Top 5 Genres by Movie Count")

            geners_df=pd.read_csv(r"C:\Users\nagab\OneDrive\Desktop\ml_1\ml(movie)\tmdb_generse.csv")

            genre_counts=geners_df['name'].value_counts().head(5)

            # Plot using matplotlib
            fig, ax = plt.subplots(figsize=(5,2))
            genre_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_xlabel("Genre")
            ax.set_ylabel("Number of Movies")
            ax.set_title("Top 5 Genres by Movie Count")
            st.pyplot(fig)

    with col2:
        if st.button("🎯 Profit Distribution"):

            movies_df=pd.read_csv(r"C:\Users\nagab\OneDrive\Desktop\ml_1\ml(movie)\tmbd_movies_df.csv")
            plt.figure(figsize=(10, 6))
            sle.histplot(movies_df['profit'], bins=50, kde=True)
            plt.title("Profit Distribution")
            plt.xlabel("Profit")
            plt.ylabel("Number of Movies")
            plt.show()
            st.pyplot(plt)

    




    
    





    

   
