# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time





def get_book(url):
    time.sleep(0.5)
    text = requests.get(url).text
    text = re.findall(r'.*\*\*\*.*START.*\*\*\*(.*)\*\*\*.*END.*\*\*\*.*', text, re.DOTALL)[0]
    text = re.sub('\r\n', '\n', text)
    return text





def tokenize(book_string):
    strip = book_string.strip()
    lst = re.findall(r"\w+|\S", re.sub(r"[\n]{2,}", "\x03\x02", strip))
    return ["\x02"] + lst + ["\x03"]





class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        unique = np.unique(tokens)
        return pd.Series(index=unique, data=1 / len(unique))
    
    def probability(self, words):
        return np.prod(np.array([self.mdl.get(word, 0) for word in words])) 
        
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, p=self.mdl.values, size=M))





class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        return pd.Series(index=tokens,data=tokens).value_counts(normalize=True)
    
    def probability(self, words):
        return np.prod(np.array([self.mdl.get(word, 0) for word in words])) 
        
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, p=self.mdl.values, size=M))





class NGramLM(object):
    
    def __init__(self, N, tokens):
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        return [tuple(tokens[j] for j in range(i, i + self.N)) for i in range(0, len(tokens) - self.N + 1)]
        
    def train(self, ngrams):
        # N-Gram counts C(w_1, ..., w_n)
        ngram_counts = pd.Series(ngrams).value_counts()
        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        n1grams = [ngram[: self.N - 1] for ngram in ngrams]
        n1gram_counts = pd.Series(n1grams).value_counts()

        # Create the conditional probabilities using above
        probabilities = [ngram_count / n1gram_counts[ngram[:self.N - 1]] for (ngram, ngram_count) in ngram_counts.items()]
        
        # Return DataFrame with above info
        return pd.DataFrame({
            'ngram': ngram_counts.index,
            'n1gram': [ngram[:self.N - 1] for ngram in ngram_counts.index],
            'prob': probabilities
        })
    
    def probability(self, words):
        prob = 1.0
        n = len(words)

        for i in range(n):
            # get words/ngram we want to look at
            ngram = tuple(words[j] for j in range(i - self.N + 1, i + 1) if j >= 0)

            # get correct LM
            lm = self
            for _ in range(self.N - len(ngram)):
                lm = lm.prev_mdl
            
            if i == 0: # Unigram model
                prob *= lm.probability(ngram)
            else: # NGram model
                lm_mdl = lm.mdl
                if (lm_mdl["ngram"] == ngram).sum() > 0:
                    prob *= lm_mdl[lm_mdl["ngram"] == ngram]["prob"].iloc[0]
                else:
                    prob = 0


        return prob

    def sample(self, M):
        # Helper function to generate sample tokens with given len (excluding '\x02')
        def sample_tokens(length):
            if length == 0:
                return []
            
            tokens = ['\x02']
            for i in range(length):

                # get correct LM
                lm = self
                for _ in range(self.N - i - 2):
                    lm = lm.prev_mdl
                
                n1gram = tuple(tokens if len(tokens) < self.N else tokens[-self.N + 1:])
                sample_space = lm.mdl[lm.mdl['n1gram'] == n1gram]
                tokens.append(np.random.choice(sample_space['ngram'], p=sample_space['prob'])[-1])
            
            return tokens

        # Transform the tokens to strings
        return ' '.join(sample_tokens(M - 1) + ['\x03'])
