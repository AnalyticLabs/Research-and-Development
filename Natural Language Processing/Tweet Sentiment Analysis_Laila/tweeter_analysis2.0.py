# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 21:23:54 2019

@author: 91953
"""

#%% Importing the libraries

import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
import seaborn
import itertools
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from wordcloud import  STOPWORDS
from sklearn import ensemble
from sklearn import tree
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.util import mark_negation
from matplotlib import pylab
pylab.rcParams['figure.figsize'] = (15, 9)

import warnings
warnings.filterwarnings("ignore")

#%% Functions
def clean_tweet(tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet)).split())

def get_tweet_sentiment(tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
        
def tweet_clean(df):
    temp_df = df.copy()
    # Remove hyperlinks
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('https?:\/\/.*\/\w*', '', regex=True)
    # Remove citations
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('\@\w*', '', regex=True)
    # Remove tickers
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('\$\w*', '', regex=True)
    # Remove punctuation
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('[' + string.punctuation + ']+', '', regex=True)
    # Remove quotes
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('\&*[amp]*\;|gt+', '', regex=True)
    # Remove RT
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('RT', '', regex=True)
    # Remove linebreak, tab, return
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('[\n\t\r]+', ' ', regex=True)
    # Remove via with blank
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('via+\s', '', regex=True)
    # Remove multiple whitespace
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('\s+\s+', ' ', regex=True)
    # Remove multiple whitespace
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('\s+\s+', ' ', regex=True)
    # Remove HashTags 
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('\#+[\w_]+[\w\'_\-]*[\w_]+', ' ', regex=True)
    # Remove Smileys
    temp_df.loc[:, "Text"] = temp_df.loc[:, "Text"].replace('[:=]+(|o|O| )+[D\)\]]+[\(\[]+[pP]+[doO/\\]+[\(\[]+(\^_\^|)', ' ', regex=True)
    # Remove empty rows
    temp_df = temp_df.dropna()
    return temp_df

def regularExpression(textToFilter):
    filteredTweet = []
    retweetPattern = 'RT|@RT'
    urlPattern = 'https://[a-zA-Z0-9+&@#/%?=~_|!:,.;]*'

    for textLine in textToFilter:
        tweet = re.sub(retweetPattern,'',textLine)
        tweet = re.sub(urlPattern,'',tweet)
        filteredTweet.append(tweet)
    return filteredTweet

def nltkTokenizer(textToTokenize):
    filteredSentence = []
    usersPattern = re.compile('@[a-zA-Z0-9]*',re.UNICODE)
    hashtagPattern = re.compile('#[a-zA-Z0-9]*',re.UNICODE)
    stop_words = stopwords.words('english')
    
    for textLine in textToTokenize:
        words = re.sub(usersPattern,'',textLine)
        words = re.sub(hashtagPattern,'',words)
        words = word_tokenize(words)
        for w in words:
            if w not in stop_words and w not in '@' and w not in '#':
                filteredSentence.append(w)
    return filteredSentence

def tweet_to_words(raw_tweet):
    tweet = ''.join(c for c in raw_tweet if c not in string.punctuation)
    tweet = re.sub('((www\S+)|(http\S+))', 'urlsite', tweet)
    tweet = re.sub(r'\d+', 'contnum', tweet)
    tweet = re.sub(' +',' ', tweet)
    words = tweet.lower().split()                             
    stops = set(stopwords.words("english"))
                 
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))
    
def users(tweet):
    user = []
    usersPattern = re.compile('@[a-zA-Z0-9]*',re.UNICODE)
    
    for t in tweet:
        u = re.findall(usersPattern,t)
        user.append(u)
    return user

def split_into_tokens(Text):
    return TextBlob(Text).words

def split_into_lemmas(Text):
    Text = Text.lower()
    words = TextBlob(Text).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


def polarity_text_blob(tweet_data_df):
    tweets = tweet_data_df
    Polarity = []
    for tweet in tweets['Text']:
        Polarity.append(get_tweet_sentiment(tweet))
    return Polarity

def TextBlob_Sentiments_Plot(tweet):
    print("Ploting the TextBlob Sentiments: =======================================")
    Index = [1,2,3]
    print(tweet.Polarity.value_counts())
    plt.bar(Index,tweet.Polarity.value_counts())
    plt.xticks(Index,['neutral','positive', 'negative'],rotation=45)
    plt.ylabel('Number of Posts')
    plt.xlabel('Sentiment expressed in Posts')
    
    polar = pd.DataFrame()
    n = int(len(tweet)) 
    sen = []
    for i in range(n):
        blob = TextBlob(str(tweet[i]))
        k = blob.sentiment.polarity
        sen.append(k)
        
    polar['polarity'] = sen
    print("Printing the Polar Data Head: ==========================================")
    print(polar.head())
    
    polar.hist()
    polar.plot.line(y='polarity',figsize=(18,9))
    
    tweet_data.groupby('Polarity').describe()
    
    tweet_data['length'] = tweet_data['Text'].map(lambda text: len(text))
    print("Printing the head of Tweets Data Newly Formed: =========================")
    print(tweet_data.head())
    print("========================================================================")
    print("Histogram Plot of the Frequency: =======================================")
    tweet_data.length.plot(bins=20, kind='hist')
    
    tweet_data.hist(column='length', by='Polarity', bins=50)
    print("Printing the Shape of the Tweets: ======================================")
    print(tweet_data.shape)

def hash_tag(tweet_data):
    tweets_texts = tweet_data["Text"].tolist()
    stop_words=stopwords.words('english')
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    
    # Get all the hashhtag words that has "#"
    hashtags = ""
    for line in tweets:
        words = line.split()
        for w in words:
            if w.startswith("#"):
                hashtags += w + " " 
                
    # Get all the hashtags in a list
    hashtags_list = re.findall(r"#(\w+)", hashtags)
    return hashtags_list, hashtags

def hash_tag_wordcloud(hashtags_list,hashtags):
    print("Ploting the Hashtag WordCloud: =========================================")
    try:                           
        # Set the figure-size
        plt.figure(figsize= (20,10))
        wordcloud = WordCloud(
                              stopwords=STOPWORDS,
                              background_color='white',
                              width=3000,
                              height=2000
                             ).generate(str(hashtags_list))
        
        plt.figure(1,figsize=(20, 20))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig('./twitter_wordcloud.png', dpi=300)
        plt.show()
    except:
        print("No HashTags Found!! Please Verify the length")
    
    print(len(hashtags_list))
    print("========================================================================")

def user_name_extraction(filteredTweet):
    print("Extracting the User words used in the Tweets: ==========================")
    user = users(filteredTweet)
    
    str1 = ' '.join(str(e) for e in user)
    
    plt.figure(figsize= (20,20))
    
    wordcloud = WordCloud(width= 3000,height= 2000,background_color='white',max_words=30).generate(str1)
    plt.imshow(wordcloud)
    plt.title('Most mentioned users')
    plt.axis("off")
    plt.show()
    print("========================================================================")

def polarity_numbers_textblob(tweet_data):
    print("Ploting the Polarity Numbers Extracted from the TextBlob: ==============")
    tweet_data.Polarity.value_counts()
    
    colors=seaborn.color_palette("hls", 10) 
    pd.Series(tweet_data["Polarity"]).value_counts().plot(kind = "bar",
                            color=colors,figsize=(15,10),fontsize=20,rot = 0, title = "Total No. of Tweets for each Sentiment")
    plt.xlabel('Sentiment', fontsize=24)
    plt.ylabel('Number of Tweets', fontsize=24)
    
    colors=seaborn.color_palette("husl", 10)
    pd.Series(tweet_data["Polarity"]).value_counts().plot(kind="pie",colors=colors,
        labels=["negative", "neutral", "positive"],explode=[0.05,0.02,0.04],
        shadow=True,autopct='%.2f', fontsize=12,figsize=(12,12),title = "Total Tweets for Each Sentiment")
    
    df=tweet_data[tweet_data['Polarity']=='negative']
    words = ' '.join(df['Text'])
    cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'])
    
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=3000,
                          height=2500
                         ).generate(cleaned_word)
    
    plt.figure(1,figsize=(20, 20))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('./negative_tweet_wordcloud.png', dpi=300)
    plt.show()
    
    df=tweet_data[tweet_data['Polarity']=='positive']
    words = ' '.join(df['Text'])
    cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'])
    
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=3000,
                          height=2500
                         ).generate(cleaned_word)
    
    plt.figure(1,figsize=(20, 20))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('./positive_tweet_wordcloud.png', dpi=300)
    plt.show()
    
    df=tweet_data[tweet_data['Polarity']=='neutral']
    words = ' '.join(df['Text'])
    cleaned_word = " ".join([word for word in words.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=3000,
                          height=2500
                         ).generate(cleaned_word)
    plt.figure(1,figsize=(20, 20))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    print("========================================================================")

def test_train_set_creation(tweet_data):
    tweet_data['clean_tweet']=tweet_data['Text'].apply(lambda x: tweet_to_words(x))
    train,test = train_test_split(tweet_data,test_size=0.33,random_state=0)
    return train, test

def classifiers():
    Classifiers = [
        LogisticRegression(C=0.001,multi_class='multinomial',max_iter=10,solver='sag', tol=1e-1),
        
        RandomForestClassifier(n_estimators=200, bootstrap=True, class_weight=None, criterion='gini',
                max_depth=50, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_jobs= -1,
                oob_score=False, random_state=10),
        
        AdaBoostClassifier(n_estimators=100, random_state=10),
        
        BernoulliNB(),
        
        MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
        
        KNeighborsClassifier(algorithm='auto', metric='minkowski',
               metric_params=None, n_neighbors=2, p=2,
               weights='uniform'),
        
        tree.DecisionTreeClassifier(),
        
        ensemble.ExtraTreesClassifier(n_estimators=100,
                                      max_features= 50,
                                      criterion= 'entropy'),
        
        ensemble.GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.001,n_estimators=50,presort='auto', random_state=None, verbose = 0)]
    return Classifiers

def countvectorizer_analizer(train_clean_tweet, test_clean_tweet, tweet_data):    
    print("Using the COUNTVECTORIZER: =============================================")    
    v = CountVectorizer(analyzer = "word")
    train_features= v.fit_transform(train_clean_tweet)
    test_features=v.transform(test_clean_tweet)
    train, test = test_train_set_creation(tweet_data)
    
    Classifiers = classifiers()
    dense_features=train_features.toarray()
    dense_test= test_features.toarray()
    Accuracy=[]
    Model=[]
    print("Entering into the Classifiers: =========================================")
    for classifier in Classifiers:
        try:
            fit = classifier.fit(train_features,train['Polarity'])
            pred = fit.predict(test_features)
        except Exception:
            fit = classifier.fit(dense_features,train['Polarity'])
            pred = fit.predict(dense_test)
        accuracy = accuracy_score(pred,test['Polarity'])
        print("====================================================================")
        print("********************************************************************")
        print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))
        print("********************************************************************")
        print("====================================================================")
        Accuracy.append(accuracy)
        print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(test['Polarity'], pred)))
        cm = metrics.confusion_matrix(test['Polarity'], pred)
        print("--------------------------------------------------------------------")
        print("Confusion matrix:\n%s" % cm)
        print("--------------------------------------------------------------------")
        Model.append(classifier.__class__.__name__)
        print("********************************************************************")
        print("====================================================================")
        
    print("Ploting the Model Performances: ========================================")
    Index = [1,2,3,4,5,6,7,8,9]
    plt.figure(1,figsize=(20, 10))
    font = {'weight' : 'bold',
            'size'   : 25}
    
    plt.rc('font', **font)
    
    plt.bar(Index,Accuracy)
    plt.xticks(Index, Model,rotation=45)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.title('Accuracies of Models')

def BOWTransformer(tweet_data):
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(tweet_data['Text'])
    return bow_transformer

def TFIDF_Transformer(tweets_bow):
    tfidf_transformer = TfidfTransformer().fit(tweets_bow)
    tfidf1 = tfidf_transformer.transform(bow1)
    return tfidf1

def CountVectorizer_original_data(tweet_data):
    print("Using the CountVectorizer on the Original Data: ========================")
    tweets = tweet_data['Text']
    cv = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=.95, stop_words='english')
    bow = cv.fit_transform(tweets)
    
    # use below if you need a data frame
    
    X, Y = bow, (tweet_data['Polarity']).ravel()
    
    binarize = Binarizer()
    X = binarize.fit_transform(X)
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y,test_size=0.3)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(preds, y_test)
    print("========================================================================")
    print("************************************************************************")
    print('Accuracy is ' + str(accuracy))
    print("------------------------------------------------------------------------")
    #print("Classification report for classifier %s:\n%s\n"
    #      % (classifier, metrics.classification_report(y_test, preds)))
    cm = metrics.confusion_matrix(y_test, preds)
    print("------------------------------------------------------------------------")
    print("Confusion matrix:\n%s" % cm)
    print("========================================================================")
    
    X, Y = bow, (tweet_data['Polarity']).ravel()
    ss = StandardScaler()
    X = X.toarray()
    X = ss.fit_transform(X)
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y,test_size=0.3)
    
    model = SVC()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    accuracy = accuracy_score(preds, y_test)
    print("************************************************************************")
    print('Accuracy is '+str(accuracy))
    print("------------------------------------------------------------------------")
    #print("Classification report for classifier %s:\n%s\n"
    #      % (classifier, metrics.classification_report(y_test, preds)))
    cm = metrics.confusion_matrix(y_test, preds)
    print("------------------------------------------------------------------------")
    print("Confusion matrix:\n%s" % cm)
    print("========================================================================")
    
    models = [('mNB' , MultinomialNB()),
              ('bNB' , BernoulliNB()),
              ('svc' , SVC())]
    
    print('{0}\t{1:<1}\t{2:<4}\t{3:<4}'.format("ACCURACY", "MEAN", "MIN", "MAX"))
    
    for name, model in models:    
        X, Y = bow, (tweet_data['Polarity']).ravel()
        
        if name == 'bNB':
            binarize = Binarizer()
            X = binarize.fit_transform(X)
        elif name == 'svc':
            ss = StandardScaler()
            X = X.toarray()
            X = ss.fit_transform(X)
            
        cv = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
        
        print('{0}\t{1:<3}\t{2:<4}\t{3:<4}'.format(name, round(cv.mean(), 4), round(cv.min(), 4), round(cv.max(), 4)))

def ModelRun_TFIDF(tweets_tfidf, tweet_data):
    print("************************************************************************")
    print("Running Multinomial NB on TF-IDF")
    print("************************************************************************")
    polarity_detector = MultinomialNB().fit(tweets_tfidf, tweet_data['Polarity'])
    
    print('predicted:', polarity_detector.predict(tfidf1[0]))
    print('expected:', tweet_data.Polarity[0])
    
    all_predictions = polarity_detector.predict(tweets_tfidf)
    print(all_predictions)
    
    print('accuracy', accuracy_score(tweet_data['Polarity'], all_predictions))
    print('confusion matrix\n', confusion_matrix(tweet_data['Polarity'], all_predictions))
    print('(row=expected, col=predicted)')
    
    plt.figure(figsize=(10,10))
    plt.matshow(confusion_matrix(tweet_data['Polarity'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    
    print(classification_report(tweet_data['Polarity'], all_predictions))
    print("========================================================================")
    
    print("************************************************************************")
    print("Running Support Vector Machines on TF-IDF")
    print("************************************************************************")
    polarity_detector = SVC().fit(tweets_tfidf, tweet_data['Polarity'])
    print('predicted:', polarity_detector.predict(tfidf1[0]))
    print('expected:', tweet_data.Polarity[0])
    
    all_predictions = polarity_detector.predict(tweets_tfidf)
    print(all_predictions)
    
    print('accuracy', accuracy_score(tweet_data['Polarity'], all_predictions))
    print('confusion matrix\n', confusion_matrix(tweet_data['Polarity'], all_predictions))
    print('(row=expected, col=predicted)')
    
    plt.matshow(confusion_matrix(tweet_data['Polarity'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    
    print("========================================================================")
    print("Comparison Run: ========================================================")
    
    X_train, X_test, y_train, y_test = \
        train_test_split(tweet_data['Text'], tweet_data['Polarity'], test_size=0.2)
    
    vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf=True,stop_words='english')
    train_corpus_tf_idf = vectorizer.fit_transform(X_train) 
    test_corpus_tf_idf = vectorizer.transform(X_test)
    
    svm_model = LinearSVC()
    nb_model = MultinomialNB()
    
    svm_model.fit(train_corpus_tf_idf,y_train)
    nb_model.fit(train_corpus_tf_idf,y_train)
    
    svm_result = svm_model.predict(test_corpus_tf_idf)
    nb_result = nb_model.predict(test_corpus_tf_idf)
    
    print('accuracy', accuracy_score(y_test, svm_result))
    print('confusion matrix\n', confusion_matrix(y_test, svm_result))
    print('(row=expected, col=predicted)')
    plt.matshow(confusion_matrix(y_test, svm_result), cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    
    print('accuracy', accuracy_score(y_test, nb_result))
    print('confusion matrix\n', confusion_matrix(y_test, nb_result))
    print('(row=expected, col=predicted)')
    plt.matshow(confusion_matrix(y_test, nb_result), cmap=plt.cm.binary, interpolation='nearest')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')
    
    clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       tokenizer=word_tokenize,         # ! Comment line to include mark_negation and uncomment next line
                                       #tokenizer=lambda text: mark_negation(word_tokenize(text)), 
                                       preprocessor=lambda text: text.replace("<br />", " "),
                                       max_features=10000) ),
        ('classifier', LinearSVC())
    ])
     
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    
     
    clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       #tokenizer=word_tokenize,         # ! Comment line to include mark_negation and uncomment next line
                                       tokenizer=lambda text: mark_negation(word_tokenize(text)), 
                                       preprocessor=lambda text: text.replace("<br />", " "),
                                       max_features=10000) ),
        ('classifier', LinearSVC())
    ])
     
    clf.fit(X_train, y_train)
    t = clf.score(X_test, y_test)

    print("Analysis for the Linear SVC using CountVectorizer: ", t)
    print("========================================================================")
    print("Running the N-Grams on the Tweets: ")
    print("========================================================================")
    bigram_clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(2, 2),
                                       tokenizer=word_tokenize, 
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "),)),
        ('classifier', LinearSVC())
    ])
     
    bigram_clf.fit(X_train, y_train)
    t = bigram_clf.score(X_test, y_test)
    print("Bi Gram Analysis Results with Ngram 2,2: ", t)
    print("------------------------------------------------------------------------")
    unigram_bigram_clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       tokenizer=word_tokenize,
                                       # tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "),)),
        ('classifier', LinearSVC())
    ])
     
    unigram_bigram_clf.fit(X_train, y_train)
    t = unigram_bigram_clf.score(X_test, y_test)
    print("UNigram Bigram Analysis with ngram 1,2: ", t)
    print("------------------------------------------------------------------------")
    unigram_bigram_clf = Pipeline([
        ('vectorizer', CountVectorizer(analyzer="word",
                                       ngram_range=(1, 2),
                                       #tokenizer=word_tokenize,
                                       tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                       preprocessor=lambda text: text.replace("<br />", " "),)),
        ('classifier', LinearSVC())
    ])
     
    unigram_bigram_clf.fit(X_train, y_train)
    t = unigram_bigram_clf.score(X_test, y_test)
    print("UNigram Bigram Analysis with ngram 1,2 and Tokennization: ", t)
    print("------------------------------------------------------------------------")
    print("END of the Analysis")
    print("------------------------------------------------------------------------")
    print("========================================================================")
    
def main():
    file_path = input("Please Enter the File Path: ")
    tweets = pd.read_csv(file_path)
    
    print(tweets.shape[0])
    
    tweet = tweets["Text"]
    
    cleaned_Tweet = clean_tweet(tweet)
    
    Polarity = polarity_text_blob(tweets)
    tweet = tweets['Text']
    data_sent = {'Text': tweet, 'Polarity': Polarity}
    
    tweet_data = pd.DataFrame(data=data_sent)
    
    print("========================================================================")
    print("Printing the Head of the Tweets: =======================================")
    print(tweet_data.head())
    print("========================================================================")

    TextBlob_Sentiments_Plot(tweet_data)
    
    hashtags_list, hashtags = hash_tag(tweet_data)
    
    hash_tag_wordcloud(hashtags_list, hashtags)

    # Text contains 'RT' for every retweet and url refrences 
    # We want to remover 'RT' and URL
    
    filteredTweet = regularExpression(tweet_data.Text)
    filteredSentence = nltkTokenizer(filteredTweet)
    
    
    hashtagList = list(itertools.chain.from_iterable(hashtags_list))
    hashtagCount = {}
    
    for h in hashtagList:
        if h in hashtagCount:
            hashtagCount[h] +=1
        else:
            hashtagCount[h] = 1
            
    # Extracting hastags that occurs more than 1000 times
    
    hashtagCount = { k : v for k,v in hashtagCount.items() if v >10}
    name = [k for k in hashtagCount if k ]
    value = [v for v in hashtagCount.values()]

    user_name_extraction(filteredTweet)

    
    polarity_numbers_textblob(tweet_data)

    
    print("========================================================================")
    print("Starting the Machine Learning Analysis: ================================")
    print("========================================================================")
    
    train, test = test_train_set_creation(tweet_data)
    
    
    train_clean_tweet=[]
    for tweet in train['clean_tweet']:
        train_clean_tweet.append(tweet)
    test_clean_tweet=[]
    for tweet in test['clean_tweet']:
        test_clean_tweet.append(tweet)
    
    
    countvectorizer_analizer(train_clean_tweet, test_clean_tweet)
    
    CountVectorizer_original_data(tweet_data)
    
    
    
    
    tweet1 = tweet_data['Text'][0]
    print(tweet1)
    
    tweet_data.Text.head()
    
    tweet_data.Text.apply(split_into_tokens)
    
    
    
    bow_transformer = BOWTransformer(tweet_data)
    
    print("Printing the COuntVectorizer BOW: ======================================")
    print(len(bow_transformer.vocabulary_))
    print("========================================================================")
    
    bow1 = bow_transformer.transform([tweet1])
    print(bow1)
    print(bow1.shape)
    
    print("========================================================================")
    tweets_bow = bow_transformer.transform(tweet_data['Text'])
    print('sparse matrix shape:', tweets_bow.shape)
    print('number of non-zeros:', tweets_bow.nnz)
    print('sparsity: %.2f%%' % (100.0 * tweets_bow.nnz / (tweets_bow.shape[0] * tweets_bow.shape[1])))
    print("========================================================================")
    
    
    tfidf1 = TFIDF_Transformer(tweets_bow)
    
    print("Printing the TF-IDF Vectors: ===========================================")
    print(tfidf1)
    print("========================================================================")
    print("TF-IDF Shape ===========================================================")
    print(tfidf1.shape)
    print("========================================================================")
    tweets_tfidf = tfidf1#_transformer.transform(tweets_bow)
        
    ModelRun_TFIDF(tweets_tfidf, tweet_data)    

#%% Loading the Dataset
if __name__ == "__main__":
    main()