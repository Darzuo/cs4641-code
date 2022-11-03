import re
import nltk
import numpy as np
import os
import email.policy
from email import parser
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

os.listdir('../input/hamnspam/')

def load_email(is_spam, filename):
    directory = "../input/hamnspam/spam" if is_spam else "../input/hamnspam/ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return parser.BytesParser(policy=email.policy.default).parse(f)

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures

def html_to_plain(email):
    try:
        soup = BeautifulSoup(email.get_content(), 'html.parser')
        return soup.text.replace('\n\n','')
    except:
        return "empty"

def email_to_plain(email):
    struct = get_email_structure(email)
    for part in email.walk():
        partContentType = part.get_content_type()
        if partContentType not in ['text/plain','text/html']:
            continue
        try:
            partContent = part.get_content()
        except: # in case of encoding issues
            partContent = str(part.get_payload())
        if partContentType == 'text/plain':
            return partContent
        else:
            return html_to_plain(part)

class EmailToWords(BaseEstimator, TransformerMixin):
    def __init__(self, stripHeaders=True, lowercaseConversion = True, punctuationRemoval = True, 
                 urlReplacement = True, numberReplacement = True, stemming = True):
        self.stripHeaders = stripHeaders
        self.lowercaseConversion = lowercaseConversion
        self.punctuationRemoval = punctuationRemoval
        self.urlReplacement = urlReplacement
        self.numberReplacement = numberReplacement
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_to_words = []
        for email in X:
            text = email_to_plain(email)
            if text is None:
                text = 'empty'
            if self.lowercaseConversion:
                text = text.lower()
            if self.punctuationRemoval:
                text = text.replace('.','')
                text = text.replace(',','')
                text = text.replace('!','')
                text = text.replace('?','')
            if self.urlReplacement:
                text = re.sub(r'http\S+', 'url', text)
            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_count = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_count[stemmed_word] += count
                word_counts = stemmed_word_count
            X_to_words.append(word_counts)
        return np.array(X_to_words)

class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += min(count, 10)
        self.most_common = total_word_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(self.most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

# Counts the number of occurrences of emails with the following pre-processing:
# - Strip email headers
# - Convert to lowercase
# - Remove punctuation
# - Replace urls with "url"
# - Perform Stemming (trim word endings with library)
# Given:
#   - emails : list of emails to be counted
# Returns:
#   - count_array : list of occurrences of each processed word in the email, index 0 = total count, index 1 = first word
#   - vocab_array : vocab list to act as a dictionary for count_array in the form {'word': index} with index starting at 1

def count_occurrences(emails):
    wordCounts = EmailToWords().fit_transform(emails)
    vocab_transformer = WordCountToVector()
    count_vector = vocab_transformer.fit_transform(wordCounts)
    count_array = count_vector.toarray()
    vocab_array = vocab_transformer.vocabulary_
    return (count_array, vocab_array) 
    
ham_filenames = [name for name in sorted(os.listdir('../input/hamnspam/ham')) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir('../input/hamnspam/spam')) if len(name) > 20]
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

emails = spam_emails[2:3]

count_array, vocab_array = count_occurrences(emails)

print(count_array)
print(vocab_array)