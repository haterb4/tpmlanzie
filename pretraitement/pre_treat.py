import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from .cleanning import cleaning
import pickle
stop_words = stopwords.words("english")

def getTokenizedWord(post):
    post_df = pd.DataFrame({"comment": [post]})
    post_df.rename(columns={'comment':'text'}, inplace = True)
    clean_post_df = post_df['text'].apply(cleaning)
    clean_post_df = pd.DataFrame(clean_post_df)  
    clean_post_df['no_sw'] = clean_post_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    frequent_word_counter = pickle.load(open(os.path.join("predictors", "frequent_word_counter.amod",), "rb"))
    FREQWORDS = set([w for (w, wc) in frequent_word_counter.most_common(10)])
    remove_freqwords = lambda text: " ".join([word for word in str(text).split() if word not in FREQWORDS])
    clean_post_df["wo_stopfreq"] = clean_post_df["no_sw"].apply(lambda text: remove_freqwords(text))
    wordnet_lem = WordNetLemmatizer()
    clean_post_df['wo_stopfreq_lem'] = clean_post_df['wo_stopfreq'].apply(wordnet_lem.lemmatize)
    clean_post_changed_df = clean_post_df.drop(columns=['text','no_sw', 'wo_stopfreq'])
    clean_post_changed_df.columns=['review']
    vectorizer = pickle.load(open(os.path.join("predictors", "vectorizer.amod",), "rb"))
    post_vector = vectorizer.transform(clean_post_changed_df.review)

    return post_vector