import streamlit as st
import joblib
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)


# ==========================================================
# LOAD NLTK
# ==========================================================

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(
    "models/fake_review_model.pkl"
)

tfidf = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ==========================================================
# LABEL CONVERSION
# ==========================================================

def convert_label(label):

    if label == "CG":
        return "Fake Review"

    elif label == "OR":
        return "Genuine Review"

    return "Unknown"


# ==========================================================
# HEADER
# ==========================================================

st.title("🔍 Fake Review Detection System")

st.markdown(
    """
    ### AI-Powered Product Review Analysis

    This system uses **Natural Language Processing (NLP)**
    and **Machine Learning** to classify product reviews as
    **Fake** or **Genuine**.
    """
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write("**Feature Extraction:**")
    st.write("TF-IDF")

    st.write("**Classification:**")
    st.write("Machine Learning")

    st.write("**Input:**")
    st.write("Product Review")

    st.write("**Output:**")
    st.write("Fake / Genuine")

    st.divider()

    st.info(
        "Enter a product review in the main panel "
        "and click Analyze Review."
    )


# ==========================================================
# MAIN INPUT AREA
# ==========================================================

left, right = st.columns([2, 1])


with left:

    st.subheader("📝 Enter Product Review")

    review = st.text_area(
        "Review Text",
        height=220,
        placeholder=(
            "Example:\n"
            "The product quality is excellent and "
            "I am very satisfied with my purchase."
        )
    )

    analyze = st.button(
        "🚀 Analyze Review",
        use_container_width=True
    )


with right:

    st.subheader("💡 Example")

    st.info(
        """
        **Example Review**

        The product is excellent.
        The quality is very good and
        I am completely satisfied with
        my purchase.
        """
    )

    st.warning(
        "The prediction is based on "
        "patterns learned from the training dataset."
    )


# ==========================================================
# PREDICTION
# ==========================================================

if analyze:

    if not review.strip():

        st.warning(
            "⚠️ Please enter a review first."
        )

    else:

        # Clean text
        cleaned_review = clean_text(review)

        # TF-IDF
        review_vector = tfidf.transform(
            [cleaned_review]
        )

        # Prediction
        prediction = model.predict(
            review_vector
        )[0]

        # Convert label
        result = convert_label(
            prediction
        )

        st.divider()

        st.subheader("📊 Prediction Result")


        # ==================================================
        # RESULT
        # ==================================================

        if result == "Fake Review":

            st.error(
                "⚠️ FAKE REVIEW DETECTED"
            )

            st.write(
                "The model classified this review "
                "as computer-generated."
            )

        elif result == "Genuine Review":

            st.success(
                "✅ GENUINE REVIEW"
            )

            st.write(
                "The model classified this review "
                "as an original review."
            )

        else:

            st.warning(
                "Unknown prediction."
            )


        # ==================================================
        # DETAILS
        # ==================================================

        st.subheader("🔎 Review Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Original Words",
                len(review.split())
            )

        with col2:

            st.metric(
                "Processed Words",
                len(cleaned_review.split())
            )

        with col3:

            st.metric(
                "Predicted Label",
                prediction
            )


        # ==================================================
        # PROCESSED TEXT
        # ==================================================

        with st.expander(
            "🔬 View Processed Review"
        ):

            st.write(cleaned_review)


# ==========================================================
# HOW IT WORKS
# ==========================================================

st.divider()

st.subheader("⚙️ How the System Works")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown(
        """
        ### 1️⃣ Input

        User enters a
        product review.
        """
    )

with step2:

    st.markdown(
        """
        ### 2️⃣ Preprocessing

        Text is cleaned
        and normalized.
        """
    )

with step3:

    st.markdown(
        """
        ### 3️⃣ TF-IDF

        Text is converted
        into numerical features.
        """
    )

with step4:

    st.markdown(
        """
        ### 4️⃣ Prediction

        ML model predicts
        Fake or Genuine.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Fake Review Detection System | "
    "NLP + TF-IDF + Machine Learning"
)