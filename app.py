
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# UI
st.title("📩 Email/SMS Spam Classifier")
st.write("Enter a message to check whether it is **Spam** or **Not Spam**.")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

        prob = model.predict_proba(vector_input)[0][1]
        st.write("Spam Probability:", round(prob * 100, 2), "%")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:14px;">
        © 2026 <b>Avanish Pandey</b> | 
        <a href="https://www.linkedin.com/in/avanish-pandey-976b76253/" target="_blank">
        LinkedIn
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

