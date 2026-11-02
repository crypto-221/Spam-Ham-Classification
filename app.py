import string
import pickle
import nltk
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

lr = pickle.load(open("lr.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmitizer = WordNetLemmatizer()


def remove_punc(text):
    result = ""
    for i in text:
        if i not in string.punctuation:
            result += i
    return result


def remove_digit(txt):
    new_ls = []
    for i in txt:
        if i.isalpha():
            new_ls.append(i)
    return new_ls


def remove_stop(txt):
    new_txt = []
    for i in txt:
        if i not in stop_words:
            new_txt.append(i)
    return new_txt


def lem(word):
    new_word = []
    for i in word:
        new_word.append(lemmitizer.lemmatize(i, pos="v"))
    return new_word


def Predict(msg):
    msg = remove_punc(msg)
    msg = msg.lower()
    tokens = word_tokenize(msg)
    tokens = remove_digit(tokens)
    tokens = remove_stop(tokens)
    tokens = lem(tokens)
    text = " ".join(tokens)

    vec = cv.transform([text])
    return lr.predict(vec)[0]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if message:
            result = str(Predict(message))
    return render_template("index.html", result=result, message=message)


app.run(debug= True , port =  5000)
