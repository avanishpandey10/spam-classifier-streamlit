# 📩 Email / SMS Spam Classifier

A Machine Learning web application that classifies messages as **Spam** or **Not Spam** using Natural Language Processing (NLP).
The application is built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🚀 Live Features

* Detects whether a message is **Spam** or **Not Spam**
* Text preprocessing using **NLTK**
* Feature extraction using **TF-IDF Vectorizer**
* Classification using **Naive Bayes**
* Interactive UI built with **Streamlit**

---

## 🧠 Machine Learning Workflow

Message Input
⬇
Text Preprocessing (Tokenization, Stopword Removal, Stemming)
⬇
TF-IDF Vectorization
⬇
Naive Bayes Model
⬇
Prediction (Spam / Not Spam)

---

## 🛠 Technologies Used

* Python
* Streamlit
* Scikit-Learn
* NLTK
* Pandas
* Pickle

---

## 📂 Project Structure

```
spam-classifier/
│
├── app.py              # Streamlit web application
├── train_model.py      # Script to train ML model
├── model.pkl           # Trained spam classifier model
├── vectorizer.pkl      # TF-IDF vectorizer
├── spam.csv            # Dataset used for training
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/spam-classifier.git
```

Navigate to the project folder:

```
cd spam-classifier
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

## 📊 Dataset

The model is trained on the **SMS Spam Collection Dataset**, which contains labeled SMS messages classified as **spam** or **ham**.

---

## 📌 Future Improvements

* Add Deep Learning models
* Improve UI design
* Deploy on cloud
* Add email spam detection
* Real-time API integration

---

## 👨‍💻 Author

**Avanish Pandey**

LinkedIn:
[https://www.linkedin.com/in/YOUR-LINKEDIN-USERNAME](https://www.linkedin.com/in/avanish-pandey-976b76253/)

---

## 📜 License

This project is for educational and portfolio purposes.
© 2026 Avanish Pandey
