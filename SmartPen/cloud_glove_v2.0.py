# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:07:29 2019

@author: JAE6KOR
"""
"required packages"

import numpy as np
import re
from nltk.corpus import stopwords
import pandas as pd
import scipy

def glove_scoring(s1,s2):
    
    gloveFile = r"D:\RBEI IoT HACKATHON 2019\data\glove.6B.50d.txt"
    
    def loadGloveModel(gloveFile):
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
        return model
    
    model = loadGloveModel(gloveFile)
    
    def preprocess(raw_text):

        # keep only words
        letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    
        # convert to lower case and split 
        words = letters_only_text.lower().split()
    
        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        cleaned_words = list(set([w for w in words if w not in stopword_set]))
    
        return cleaned_words

    def cosine_distance_wordembedding_method(s1, s2):

        vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
        vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
        cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
        
        return float(round(1-cosine,4))
    
    def cosine_distance_between_two_words(word1, word2):
        import scipy
        return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))
    
    glove_score = cosine_distance_wordembedding_method(s1, s2)
    
    if glove_score>=0.80:
        score1 = 5
    elif (glove_score>=0.65) & (glove_score<0.80):
        score1 = 4
    elif (glove_score>=0.50) & (glove_score<0.65):
        score1 = 3
    elif (glove_score>=0.35) & (glove_score<0.50):
        score1 = 2
    elif (glove_score>=0.20) & (glove_score<0.35):
        score1 = 1
    else:
        score1 = 0    
               
    return score1

def main():
    question = "who is the definition of science?"
    stand_ans = "Science is observing, studying and experimenting to learn how the world works.This includes the departments of learning and bodies of fact in disciplines such as anthropology, archaeology, astronomy, biology, botany, chemistry, cybernetics, geography, geology, mathematics, medicine, physics, physiology, psychology,"
    stud_ans = "Science is the study of the nature and behaviour of natural things and the knowledge that we obtain about them. ... A science is a particular branch of science such as physics, chemistry, or biology. Physics is the best example of a science which has developed strong, abstract theories."
    final_score = glove_scoring(stud_ans, stand_ans)
    print("final score:", int(final_score))
main()