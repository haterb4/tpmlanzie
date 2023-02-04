import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
#import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from cleanning import cleaning
stop_words = stopwords.words("english")

from cleanning import cleaning

def getTokenizedWord(post):
    wordDf = pd.DataFrame({"comment": [post]})
    wordDf.rename(columns={'comment':'text'}, inplace = True)
    wordDt = wordDf['text'].apply(cleaning)
    wordDt = pd.DataFrame(wordDt)
    wordDt['no_sw'] = wordDt['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    cnt = Counter()
    for text in wordDt["no_sw"].values:
        for word in text.split():
            cnt[word] += 1
    cnt.most_common(10)
    temp = pd.DataFrame(cnt.most_common(10))
    temp.columns=['word', 'count']
    
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    def remove_freqwords(text):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])
    wordDt["wo_stopfreq"] = wordDt["no_sw"].apply(lambda text: remove_freqwords(text))
    wordnet_lem = WordNetLemmatizer()
    wordDt['wo_stopfreq_lem'] = wordDt['wo_stopfreq'].apply(wordnet_lem.lemmatize)
    wordnb = wordDt.drop(columns=['text','no_sw', 'wo_stopfreq'])
    wordnb.columns=['review']
    print("tokenized word: ", wordnb)
    wordnb_tokenized_review = wordnb['review'].apply(lambda x: x.split())
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    text_count = cv.fit_transform(wordnb['review'])
    print("work")
    return True