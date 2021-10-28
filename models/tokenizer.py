import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def tokenize(text): 
    #Normalization - lowercase and punctuation removal:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    #Tokenization:
    words = text.split()
    words = word_tokenize(text)
    #Stop words removal:
    words = [w for w in words if w not in stopwords.words("english")]
    #Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w, pos = "v") for w in words]
    
    return lemmed