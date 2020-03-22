# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:48:39 2019

@author: JAE6KOR
"""

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
from itertools import chain    
import time
import num2words 
import re
from sklearn.feature_extraction.text import TfidfVectorizer # used to find tf-idf vector
import gensim # used for word2vec model

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
gloveFile = r"D:\NLP\Text summarization\codes\dependencies\glove.6B.50d.txt"
word2vecFile = r"D:\NLP\Text classification\word2vec pre-trained model\GoogleNews-vectors-negative300.bin"

#%%

def loadGloveModel(gloveFile):
    st = time.time()
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    print('total glove load time:'+str(round(time.time()-st,2))+" seconds")
    return model

#%%
def loadWord2vecModel(word2vecFile):
    print ("Loading Word2vec Model")
    # Load Google's pre-trained Word2Vec model.
    st = time.time()
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vecFile, binary=True) 
    print('total word2vec load time:'+str(round(time.time()-st,2))+" seconds") 
    return model

#%%
model_glove = loadGloveModel(gloveFile)
model_word2vec = loadWord2vecModel(word2vecFile)
#%%
def preprocess_sentence(sentence, stop_words):

#    print(sentence)
    sentence = word_tokenize(sentence)
    sentence = [word for word in sentence if word not in string.punctuation]
    sentence = list(chain.from_iterable([w.split('-') for w in sentence]))
    sentence = list(chain.from_iterable([w.split("'") for w in sentence]))
    sentence = [word.lower() for word in sentence]
    sentence = [w for w in sentence if w not in stop_words]
    sentence = [lemmatizer.lemmatize(word) for word in sentence]
    sentence1 = []
    for word in sentence:
        find_num = re.findall(r'[0-9]+',word)
        if len(find_num)>0:
            if len(find_num[0])==len(word):
                sentence1+=[num2words.num2words(word)]
        else:
            sentence1+=[word]
            
    sentence1 = list(chain.from_iterable([w.split(" ") for w in sentence1]))
    sentence1 = [word for word in sentence1 if word.isalpha()]
    return sentence1


def sentence_similarity(sent1, sent2, method, stop_words):
    if method=="glove":
        full_vect_1 = []
        full_vect_2 = []
        for word in preprocess_sentence(sent1, stop_words):
            try:
                full_vect_1+=[model_glove[word]]
            except:
                print(word)
        for word in preprocess_sentence(sent2, stop_words):
            try:
                full_vect_2+=[model_glove[word]]
            except:
                print(word)                
        vector_1 = np.mean(full_vect_1,axis=0)
        vector_2 = np.mean(full_vect_2,axis=0)
        return 1-cosine_distance(vector_1, vector_2)
    elif method=="word2vec":
        full_vect_1 = []
        full_vect_2 = []
        for word in preprocess_sentence(sent1, stop_words):
            try:
                full_vect_1+=[model_word2vec[word]]
            except:
                print(word)
        for word in preprocess_sentence(sent2, stop_words):
            try:
                full_vect_2+=[model_word2vec[word]]
            except:
                print(word)                
        vector_1 = np.mean(full_vect_1,axis=0)
        vector_2 = np.mean(full_vect_2,axis=0)
        return 1-cosine_distance(vector_1, vector_2)
    elif method=='countvectorizer':
        preprocessed_sent1 = preprocess_sentence(sent1, stop_words)
        preprocessed_sent2 = preprocess_sentence(sent2, stop_words)
        all_words = list(set(preprocessed_sent1 + preprocessed_sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        # build the vector for the first sentence (approach: count vectorizer)
        for w in preprocessed_sent1:
            vector1[all_words.index(w)] += 1
     
        # build the vector for the second sentence
        for w in preprocessed_sent2:
            vector2[all_words.index(w)] += 1
        return 1 - cosine_distance(vector1, vector2)
    elif method=='tfidfvectorizer':
        preprocessed_sent1 = ' '.join(preprocess_sentence(sent1, stop_words))
        preprocessed_sent2 = ' '.join(preprocess_sentence(sent2, stop_words))
        
        tfidfvectorizer = TfidfVectorizer()
        X = tfidfvectorizer.fit_transform([preprocessed_sent1, preprocessed_sent2]).toarray()
#        print(tfidfvectorizer.get_feature_names())

        vector1 = X[0]
        vector2 = X[1]
        
        return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, method, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], method, stop_words)
    return similarity_matrix


def generate_summary(file_name, method,top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    file = open(file_name, "r")
    filedata = file.readlines()
    article = [x for x in filedata[0].split(".") if len(x)>0]
    
    sentences = article
    org_sentences = article

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, method, stop_words)
#    print(sentence_similarity_martix)
#    print(sentence_similarity_martix)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(org_sentences)), reverse=True)    
#    print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    
    summarize_text = [ranked_sentence[i][1] for i in range(top_n)]
    
    summarize_text = ". ".join(summarize_text)
    
    print(summarize_text)
# let's begin
filepath=r"D:\NLP\Text summarization\codes"
model_names=['glove','word2vec','countvectorizer', 'tfidfvectorizer']
for model_name in model_names[:]:
    print('\n'+model_name)
    generate_summary(os.path.join(filepath,"msft.txt"),method=model_name, top_n=3)
