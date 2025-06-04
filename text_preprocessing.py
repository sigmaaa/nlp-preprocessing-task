from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re


def cleanup_text(text_array):
    clean_arr = np.array([clean_string(word) for word in text_array])
    stop_words = set(stopwords.words('english'))
    clean_arr = np.array(
        [word for word in clean_arr if word and word not in stop_words])
    return clean_arr


def clean_string(s):
    s = s.lower()
    return re.sub(r'[^a-zA-Z]+', '', s) if isinstance(s, str) else s


def stemming(text_array):
    ps = PorterStemmer()
    return np.array([ps.stem(word) for word in text_array])


def term_extraction(text_array):
    clean_array = cleanup_text(text_array)
    return stemming(clean_array)
