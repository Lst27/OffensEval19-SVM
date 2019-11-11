import math
import nltk
from nltk.corpus import stopwords
from gensim.models import FastText

class invertedIndex:
    index = {}
    lex = {}
    n_docs = 0
    max_len = 0

    def init(self, filename):
        cnt = 0
        with open(filename, "r", encoding="utf8") as f:
            f.readline()
            for line in f:
                parts = line.split('\t')
                if len(parts) != 5:
                    continue
                self.n_docs += 1
                cur_len = 0
                for word in nltk.word_tokenize(parts[1]):
                    cur_len += 1
                    word = word.lower()
                    if not self.lex.__contains__(word):
                        self.lex[word] = cnt
                        cnt += 1
                    if self.index.__contains__(word):
                        postings = self.index.get(word)
                        try:
                            postings.index(int(parts[0]))
                        except ValueError:
                            postings.append(int(parts[0]))
                        #binary search python?
                    else:
                        self.index[word] = [int(parts[0])]
                if cur_len > self.max_len:
                    self.max_len = cur_len
        return self.index


class Sentiment:
    sentiments = {}
    sent_lex = {}

    def __init__(self, path):
        cnt = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                self.sentiments[parts[0]] = float(parts[1])
                self.sent_lex[parts[0]] = cnt
                cnt += 1
            self.n_words = cnt

class VecStuff:
    idf_table = {}
    sent_table = {}
    senti_len = {}
    lexicon = {}
    n_docs = 0
    embeds = FastText
    max_len = 0


    def __init__(self, path):
        print("")

    def load(self, path):
        print("loading resources, this may take some minutes ... ")
        x = invertedIndex()
        self.idf_table = x.init(path)
        self.lexicon = x.lex
        self.n_docs = x.n_docs
        self.max_len = x.max_len
        y = Sentiment('../vader.txt')
        self.sent_table = y.sentiments
        self.senti_lex = y.sent_lex
        self.embeds = FastText.load_fasttext_format("../cc.en.300.bin")
        self.stopwords = set(stopwords.words('english'))


    def get_tf_idf_vec(self, tweet):
        sent_map = {}
        for word in nltk.word_tokenize(tweet):
            word = word.lower()
            if sent_map.__contains__(word):
                n = sent_map.get(word)
                sent_map[word] = n+1
            else:
                sent_map[word] = 1
        tf_idf_vec = [0] * (len(self.lexicon) + 1)
        for word in nltk.word_tokenize(tweet):
            word = word.lower()
            try:
                index = self.lexicon.get(word) + 1
            except TypeError:
                index = 0
            try:
                idf_val = 1 + (self.n_docs / len(self.idf_table.get(word)))
            except TypeError:
                idf_val = 1
            tf_idf_vec[index] = math.log10(idf_val) * math.log10(1 + sent_map.get(word))
        return tf_idf_vec

    def get_positioned_tfidf(self, tweet):
        sent_map = {}
        for word in nltk.word_tokenize(tweet):
            word = word.lower()
            if sent_map.__contains__(word):
                n = sent_map.get(word)
                sent_map[word] = n + 1
            else:
                sent_map[word] = 1
        tf_idf_vec = [0] * self.max_len
        for pos, word in enumerate(nltk.word_tokenize(tweet)):
            word = word.lower()
            try:
                idf_val = 1 + (self.n_docs / len(self.idf_table.get(word)))
            except TypeError:
                idf_val = 1
            tf_idf_vec[pos] = math.log10(idf_val) * math.log10(1 + sent_map.get(word))
        return tf_idf_vec

    def get_combined_vec(self, tweet):
        tf_idf_vec = self.get_tf_idf_vec(tweet)
        sent_vec = [0] * len(self.senti_lex)
        emb = [0] * 300
        for word in nltk.word_tokenize(tweet):
            word = word.lower()
            if word in self.stopwords:
                continue
            try:
                index = self.senti_lex[word]
                sent_vec[index] = self.sent_table[word]
            except:
                index = 0
            try:
                emb += self.embeds[word]
            except:
                continue
        return tf_idf_vec + sent_vec + emb.tolist()

    def get_positioned_combined_vec(self, tweet):
        tf_idf_vec = self.get_positioned_tfidf(tweet)
        sent_vec = [0] * self.max_len
        emb = [0] * 300
        for pos, word in enumerate(nltk.word_tokenize(tweet)):
            word = word.lower()
            if word in self.stopwords:
                continue
            try:
                sent_vec[pos] = self.sent_table[word]
            except:
                sent_vec[pos] = 0
            try:
                emb += self.embeds[word]
            except:
                continue
        return tf_idf_vec + sent_vec + emb.tolist()

    def get_emb_vec(self, tweet):
        emb = [0] * 300
        for word in nltk.word_tokenize(tweet):
            word = word.lower()
            if word in self.stopwords:
                continue
            try:
                emb += self.embeds[word]
            except:
                continue
        return emb.tolist()

