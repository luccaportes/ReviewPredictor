from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle
import os
import unidecode
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class ReviewPredictor:
    def __init__(self, class_type="SVM"):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if class_type == "SVM":
            with open(current_dir + "/binary_files/svm.pickle", "rb") as f:
                self.classifier = pickle.load(f)
        elif class_type == "RandomForest":
            with open(current_dir + "/binary_files/random_forest.pickle", "rb") as f:
                self.classifier = pickle.load(f)
        else:
            raise ValueError("Incorrect Classifier Type")

        with open(current_dir + "/binary_files/vectorizer.pickle", "rb") as f:
            self.vectorizer = pickle.load(f)

        self.dict_transform = {3: "positive", 1: "negative"}

        self.stopw = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def predict_many(self, string_list):
        string_list = self.clean_string_list(string_list)
        vectorized = self.vectorizer.transform(string_list).toarray()
        number_answer = self.classifier.predict(vectorized)
        string_answer = [self.dict_transform[i] for i in number_answer]
        return number_answer, string_answer

    def predict_one(self, string):
        number_list, string_list = self.predict_many([string])
        return number_list[0], string_list[0]

    def clean_string_list(self, string_list):
        return [self.clean_string(string) for string in string_list]

    def clean_string(self, string):
        string = string.lower()
        # REMOVE SPECIAL CHARACTERS SUCH AS EMOJIS
        string = unidecode.unidecode(string)
        # KEEP ONLY LETTERS
        string = re.sub("[^a-zA-Z ]", "", string)
        # MAXIMUM OF 2 SEQUENTIALLY REPEATED LETTERS. I.E allllll -> all
        string = re.sub(r"(\w)\1{2,}", r"\1\1", string)
        # TRANSFORM TO LIST
        string_list = string.split()
        new_string_list = []
        # REMOVE STOPWORDS AND STEM
        for word in string_list:
            if word not in self.stopw:
                new_string_list.append(self.stemmer.stem(word))
        # CONVERT LIST BACK TO STR AND RETURN
        return " ".join(new_string_list)
