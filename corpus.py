from nltk.tokenize import RegexpTokenizer

''' for a given string of words tokenize() returns a list of words
after removing punctuation and converting words to lover case
'''


def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_text = tokenizer.tokenize(text)
    return [item.lower() for item in tokenized_text]


''' given a list of words returning the string
containing words separated by whitespace
'''


def detokenize(tokens):
    detokenized_tokens = ' '.join(tokens)
    return detokenized_tokens
