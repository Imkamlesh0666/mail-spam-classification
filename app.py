import streamlit as st
import pickle
import nltk
import time
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

# app Config
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam Message Classifier")

# bigger text box
input_sms = st.text_area("✏️ Enter the message", height=200)

if st.button('🔍 Predict'):
    start_time = time.time()

    transformed_sms = transform_text(input_sms)
    votes = []

    for i in range(7):
        model = pickle.load(open(f'model_{i}.pkl', 'rb'))
        vectorizer = pickle.load(open(f'vectorizer_{i}.pkl', 'rb'))
        vec = vectorizer.transform([transformed_sms]).toarray()
        pred = model.predict(vec)[0]
        votes.append(pred)

    final_pred = 1 if votes.count(1) >= 4 else 0
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    # show result
    st.subheader("🧾 Prediction Result:")
    if final_pred == 1:
        st.markdown("<h3 style='color: red;'>🚨 Spam</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>✅ Not Spam</h3>", unsafe_allow_html=True)

    st.info(f"🕒 Prediction Time: {elapsed_time} seconds")

    # model names that used for training also
    model_names = [
        "MultinomialNB",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "KNeighborsClassifier",
        "RandomForestClassifier",
        "SVC",
        "BernoulliNB"
    ]

    # show all votes
    vote_results = []
    for i in range(7):
        label = "Spam" if votes[i] == 1 else "Not Spam"
        color = "red" if votes[i] == 1 else "green"
        vote_results.append(f"<li><strong>{model_names[i]}</strong>: <span style='color:{color};'>{label}</span></li>")

    st.markdown("### 🗳️ Individual Model Predictions")
    st.markdown("<ul>" + "\n".join(vote_results) + "</ul>", unsafe_allow_html=True)
