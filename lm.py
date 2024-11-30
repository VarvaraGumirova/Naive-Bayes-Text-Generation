import random


class LanguageModel:
    def __init__(self, n):
        self.n = n
        self.counts = {}
        self.vocabulary = set()

    ''' train() takes a list of token sequences. Each token sequence of the list
    # is used to calculate statistics for the language model
    # After executing this function we will have a fully trained language model
    '''
    def train(self, token_sequences):
        for token_sequence in token_sequences:
            self.train_on_sequence(token_sequence)

    ''' takes a single sequence of tokens as an input and calculates statistics
    for it for training the language model
    '''
    def train_on_sequence(self, token_sequence):
        for token in token_sequence:
            self.vocabulary.add(token)
        ngrams = get_ngrams(token_sequence, self.n)
        for ngram in ngrams:
            context = ngram[0:self.n - 1]
            word = ngram[self.n - 1]
            # checking if encountered new context:
            if context not in self.counts:
                self.counts[context] = {}
            # checking if encountered new word for given context:
            if word not in self.counts[context]:
                self.counts[context][word] = 0
            self.counts[context][word] += 1

    '''p_next() takes as an input a sequence of tokens and returns
    probability distribution corresponding to the next word:
    which words can follow the given sequence of tokens and
    with which probability
'''
    def p_next(self, tokens):
        pad_symbols = [None] * (self.n - 1)
        new_tokens = pad_symbols + tokens
        context = tuple(new_tokens[-(self.n - 1):])
        if context in self.counts:
            '''if seen the context, using statistics from the
            language model to find probability of the next word
            '''
            counts_for_context = self.counts[tuple(context)]
            return normalize(counts_for_context)
        else:
            '''if not seen the context, returning
            a uniform distribution over all possible words
            in the vocabulary
            '''
            probability_uniform = 1/len(self.vocabulary)
            distribution = {word: probability_uniform for word in self.vocabulary}
            return distribution

    '''generate() method returns a sequence of words
    created by the trained language model
    '''
    def generate(self):
        list_of_tokens = []
        while True:
            distribution = self.p_next(list_of_tokens)
            generated_word = sample(distribution)
            if generated_word is None:
                '''if None was generated, stopping
                generation of the text
                and returning result
                '''
                break
            else:
                list_of_tokens.append(generated_word)
        return list_of_tokens


'''takes a dictionary of pairs 'word: count' as an input
and transforms this dictionary into a
valid probability distribution
by dividing the counts by the sum of all counts
'''


def normalize(word_counts):

    result = {}
    sum_of_counts = sum(word_counts.values())
    for key in word_counts:
        result[key] = word_counts[key] / sum_of_counts
    return result


'''takes as an input a sequence of tokens and a parameter n
which defines the n-grams and returns list of n-grams
found in the original sequence of tokens
also it pads the original sequence with None
repeated n - 1 times before and after the sequence'''
''''''


def get_ngrams(tokens, n):
    pad_symbols = [None] * (n - 1)
    new_tokens = pad_symbols + tokens + pad_symbols
    n_grams = []
    for starting_index in range(0, len(new_tokens) - n + 1):
        ending_index = starting_index + n
        current_ngram = new_tokens[starting_index: ending_index]
        n_grams.append(tuple(current_ngram))

    return n_grams


'''sample() takes as an input a distribution of the words represented
by dictionary of pairs 'word: probability'
and returns one word sample according to this distribution
'''


def sample(distribution):
    random_float = random.random()
    cumulative_prob = 0
    for key, prob in distribution.items():
        cumulative_prob += prob
        if random_float < cumulative_prob:
            return key

