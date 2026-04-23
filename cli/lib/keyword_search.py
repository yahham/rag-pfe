import string
from nltk.stem import PorterStemmer
from .search_utils import load_movies, load_stopwords

stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower() 
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text):
    text      = clean_text(text)
    stopwords = load_stopwords()
    result    = []
    def __filter(token):
        token = token.strip("\n")
        if token and token not in stopwords:
            return True
        return False 
    for token in text.split():
        if __filter(token):
            token = stemmer.stem(token)
            result.append(token)
    return result

def has_matching_token(query_tokens, movie_tokens):
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def search_command(query, n_results):
    movies = load_movies()
    result = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, movie_tokens):
            result.append(movie)
        if len(result) == n_results:
            break
    return result
