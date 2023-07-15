import re
import nltk
import spacy
from unicodedata import normalize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

def remove_html_tags(text):
    
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    return text

def stem_text(text):
    
    stem = PorterStemmer()
    tokens = tokenizer.tokenize(text)
    text = ' '.join([stem.stem(token) for token in tokens])
    return text

def lemmatize_text(text):
    
    tokens = nlp(text)
    text = [token.lemma_ for token in tokens]
    return ' '.join(text)

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile(
        '({})'.format('|'.join(contraction_mapping.keys())), 
        flags=re.IGNORECASE|re.DOTALL
        )
    def expand_match(contraction):
        match = contraction.group(0)
        contraction = contraction_mapping.get(match.lower(), match)
        return contraction
        
    text = contractions_pattern.sub(expand_match, text)
    return text

def remove_accented_chars(text):
    
    text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_chars(text, remove_digits=False):
    
    pattern = r'[^a-zA-Z0-9 ]' if not remove_digits else r'[^a-zA-Z ]'
    text = re.sub(pattern, '', text)
    return text

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    
    tokens = nltk.word_tokenize(text)
    if is_lower_case:
        tokens = [token.lower() for token in tokens]
    text = [token for token in tokens if token.lower() not in stopwords]
    return ' '.join(text)

def remove_extra_new_lines(text):
    
    text = re.sub('[\r|\n|\r\n]+', ' ', text)
    return text

def remove_extra_whitespace(text):
    
    text = re.sub('[\s]+', ' ', text)
    return text.strip()
    
def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list
):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
