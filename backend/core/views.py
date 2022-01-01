import nltk
nltk.download('punkt')
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from django.shortcuts import render
from rest_framework.views import APIView
from . models import *
from rest_framework.response import Response
from . serializer import *

# Create your views here.
  
class ReactView(APIView):
    
    serializer_class = ReactSerializer
    def parse_text(self, document):
        word_features5k_f = open("model/word_features5k.sav", "rb")
        word_features = pickle.load(word_features5k_f)
        word_features5k_f.close()
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features
    def load_model(self, file_path):
        classifier_f = open(file_path, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        return classifier

    def getPredictions(self, text):

        ONB_Clf = self.load_model('model/ONB_clf.sav')

        # Multinomial Naive Bayes Classifier
        MNB_Clf = self.load_model('model/MNB_clf.sav')


        # Bernoulli  Naive Bayes Classifier
        BNB_Clf = self.load_model('model/BNB_clf.sav')

        # Logistic Regression Classifier
        LogReg_Clf = self.load_model('model/LogReg_clf.sav')

        # Stochastic Gradient Descent Classifier
        SGD_Clf = self.load_model('model/SGD_clf.sav')
        ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)
        feats = self.parse_text(text)
        return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)
  
    def get(self, request):
        review = [ {"review": review.review} 
        for review in React.objects.all()]
        return Response(review)

    def post(self, request):
        React.objects.all().delete()
        serializer = ReactSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            prediction, confidence=self.getPredictions(request.data["review"])
            toSend={
                "prediction":prediction,
                "confidence":confidence
            }
            return  Response(toSend)

class EnsembleClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
