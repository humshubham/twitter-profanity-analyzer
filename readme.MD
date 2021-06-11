# Twitter Profanity Analyzer

Python script for checking the degree of profanity(racial slurs in this case) in tweets.

### Disclaimer : "racial_slurs.txt" contains offensive words used for analyzing tweets.  
    List created by words from ['here'](https://en.wikipedia.org/wiki/List_of_ethnic_slurs)

## Setup and Run

1. Make sure you have installed Python3 in your system

2. Download and extract or git clone this repository

3. Open up a terminal and run the following command to install required python libraries.

```python 
pip3 install pandas
pip3 install nltk

python3 -m nltk.downloader stopwords
python3 -m nltk.downloader wordnet
```

3. Run analyze.py to generate results in results.csv.

```python
python3 analyze.py
```
## Steps taken in the process

1. Create a list of slurs or offensive words

2. Get twitter data

3. Clean tweets by removing punctuation, stopwords, stemming & lemmitization

4. Calculate degree of profanity which is given by :
### Degree of Profanity = sum(number of profane words in the sentence)/ total number of words
