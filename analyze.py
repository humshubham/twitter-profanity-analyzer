import json 
import string
from types import resolve_bases
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
 
#read file containing racial slurs
def get_profane(file):
    profane=[]
    f=open(file,'r+')
    profanity = f.readlines()
    for each in profanity:
        profane.append(each.strip())
    return profane

#read file containing twitter data
def get_tweets(file):
    with open(file,'r+') as f:
	    twitterData = json.load(f)
    tweetInfo = pd.DataFrame(twitterData['info'])
    return tweetInfo

#function to remove punctation marks()e.g ?,!, etc)
def remove_punctuation(text):
    no_punctuation = "".join([c for c in text if c not in string.punctuation])
    return no_punctuation

#function to remove stopwords like is, an, the, etc
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

#function for generating root form of the words
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(word) for word in text]
    return lem_text

def word_stemmer(text):
    stem_text = " ".join([stemmer.stem(i) for i in text])
    return stem_text

#Applying above functions to clean tweets
def clean_tweets(tweetInfo):
    tweetInfo['tweet'] = tweetInfo['tweet'].apply(lambda x: remove_punctuation(x))    
    tweetInfo['tweet'] = tweetInfo['tweet'].apply(lambda x: tokenizer.tokenize(x))    
    tweetInfo['tweet'] = tweetInfo['tweet'].apply(lambda x: remove_stopwords(x))    
    tweetInfo['tweet'] = tweetInfo['tweet'].apply(lambda x: word_lemmatizer(x))
    tweetInfo['tweet'] = tweetInfo['tweet'].apply(lambda x: word_stemmer(x))

    return tweetInfo

#calculating degree of profanity in each tweet
def calculate_degree(tweet, profane):
    degree_of_profanity = sum(1 for word in tweet if word in profane) / len(tweet)
    return degree_of_profanity

#generating final results
def calculate_profanity_degree(cleaned_tweets, profane):
    profanity_df = pd.DataFrame()
    profanity_df['id'] = tweetInfo['id']
    profanity_df['userName'] = tweetInfo['userName']
    profanity_df['tweet'] = cleaned_tweets['tweet'].apply(lambda x: tokenizer.tokenize(x))
    profanity_df['degree'] = profanity_df['tweet'].apply(lambda x: calculate_degree(x, profane))
    return profanity_df

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')




profane = get_profane('racial_slurs.txt')
tweetInfo = get_tweets('twitter.json')

cleaned_tweets = clean_tweets(tweetInfo)

result = pd.DataFrame()
result = calculate_profanity_degree(cleaned_tweets, profane)
result.to_csv('result.csv')

