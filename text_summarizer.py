import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np

import re, termcolor, colorama
from time import sleep

# Fix windows terminal colors
colorama.init()
# ----------------------------------------------- #

# Download stop words & tokens [Runs only one time]
# nltk.download("stopwords")
# nltk.download("punkt")
# ----------------------------------------------- #

TOP_SENTENCES = 5 # [if None, then it'll be half length of the original sentences]

class TextSummarizer:
    def __init__(self, text):
        self.stop_words = stopwords.words("english")
        self.text = text
    
    def sentence_similarity(self, sent1, sent2):
        # Tokenize words in sentences
        words1 = nltk.word_tokenize(sent1)
        words2 = nltk.word_tokenize(sent2)

        # Filter stop words & non-alphanumeric words
        words_filter = lambda w: (w.isalnum()) and (w not in self.stop_words)
        words1 = [w.lower() for w in words1 if words_filter(w)]
        words2 = [w.lower() for w in words2 if words_filter(w)]

        # Create empty vectors representing word frequency in each sentence
        all_words = list(set(words1 + words2))
        v1 = np.zeros(len(all_words))
        v2 = np.zeros(len(all_words))
        
        # Update vectors values
        for w in words1:
            v1[all_words.index(w)] += 1

        for w in words2:
            v2[all_words.index(w)] += 1
        
        # Calculate and return cosine similarity
        return (1 - cosine_distance(v1, v2))


    def build_similarity_matrix(self, sentences):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        # Populate the similarity matrix
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if (i != j):
                    similarity_matrix[i][j] = self.sentence_similarity(sentences[i], sentences[j])
        
        return similarity_matrix
    

    def generate_summary(self):
        # Tokenize sentences
        sentences = nltk.sent_tokenize(self.text)

        # Top sentences number        
        top_n = len(sentences) // 2 if (TOP_SENTENCES is None) else TOP_SENTENCES

        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentences)

        # Summarize based on similarity scores
        similarity_scores = similarity_matrix.sum(axis=1) # (axis = 1) -> sum over rows
        top_sentences_indexes = np.argsort(similarity_scores)[::-1][:top_n]
        top_sentences = np.array(sentences)[top_sentences_indexes]
        
        summarized_text = " ".join(top_sentences)

        self.summarized_text = summarized_text

    # Useless function
    def write_fancy_summary(self, total_time=2, color="light_yellow"):
        char_time = total_time / len(self.summarized_text)

        print()
        for c in self.summarized_text:
            print(termcolor.colored(c, color), end="", flush=True)
            sleep(char_time)
        print()
# ----------------------------------------------- #

if (__name__ == "__main__"):
    # User input file
    input_path = input(" File: ")
    match = re.search(r'["\'](.+)["\']', input_path)
    if (match):
        file_path = match.group(1)
    else:
        print(f"Can't specify [{input_path}]")
        input()
    
    # Read text
    with open(file_path, encoding="utf-8") as f:
        text = " ".join(f.readlines())
    
    # Summarize text
    summerizer = TextSummarizer(text)
    summerizer.generate_summary()
    summerizer.write_fancy_summary()

    input("\n Press Enter To Exist...")
# ----------------------------------------------- #
    
