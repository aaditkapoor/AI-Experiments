# Basic Naive Bayes from scratch
import os
import random
import json
from collections import Counter
import numpy as np
from typing import Dict, List

random.seed(1)

def get_p(whole:Dict, of: str = "normal"):
    """
        Get p(normal) or p(spam) depending on the condition
    """
    normal = []
    spam = []
    for k, v in enumerate(whole):
        if whole[v] == 1:
            normal.append(k)
        else:
            spam.append(k)
    if of == "normal":
        return len(normal)/len(whole)
    else:
        return len(spam)/len(whole)
            

def build_corpus(text: str):
    """
        Build corpus from text
    """
    corpus = text
    cw = Counter()
    prob = {}
    for w in corpus:
        cw[w]+=1
    for w in corpus:
        prob[w] = cw.get(w, 0) / len(corpus) # frequency/total_words
    return corpus, prob, cw
    

class NaiveBayes:
    def __init__(self, train: Dict, test: Dict):
        self.train = train
        self.normal_train_text, self.spam_train_text = NaiveBayes.get_normal_text(train)
        self.test = test
    @staticmethod
    def get_normal_text(train):
        texts: List[str] = []
        spams: List[str] = []
        for k,v in enumerate(train):
            if train[v] == 1:
                texts.extend([w.lower() for w in v.split()])
            else:
                spams.extend([w.lower() for w in v.split()])
        return texts, spams
                  
    def fit(self):
        """Train"""
        self.corpus_train_normal, self.corpus_train_normal_prob, self.normal_cw = build_corpus(self.normal_train_text)
        self.corpus_train_spam, self.corpus_train_spam_prob, self.spam_cw = build_corpus(self.spam_train_text)
        self.p_normal = get_p(self.train, "normal")
        self.p_spam = get_p(self.train, "spam")


    def predict(self, text: str, verbose=False):
        """Predict"""
        # check if the created variable exists
        if not hasattr(self, "corpus_train_normal"):
            raise ValueError("You must call .fit() before calling predict.")
        for i in text.split():
            self.p_normal = self.p_normal * self.corpus_train_normal_prob.get(i, 0) # show 0 if not found
        for i in text.split():
            self.p_spam = self.p_spam * self.corpus_train_spam_prob.get(i, 0.0) # show 0 if not found
        if not verbose:
            return 1 if self.p_normal >  self.p_spam else 0
        else:
            return "Text is normal" if self.p_normal >  self.p_spam else "Text is spam"

            

    def predict_test(self):
        """Predict with test dataset"""
        pred = []
        for i in self.test.keys():
            pred.append(self.predict(i))
        pred = np.array(pred)
        true = self.test.values()
        return (pred==true).mean()

"""
    Usage
    =========

    >>> nb = NaiveBayes(train = {"asd":1,"bf":1, "df":0}, test={})
    >>> nb.fit()
    >>> nb.predict("asd", verbose=True)
    >>> nb.predict_test
"""



        
