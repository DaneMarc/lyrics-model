from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import fasttext
import fasttext.util

class FT:
    def __init__(self):
        self.model = fasttext.load_model('wiki.simple.100.bin')
        self.stopwords = set(stopwords.words('english'))
        self.max_len = 50
        self.__len__ = len(self.model.words)

    def lyrics_to_vec(self, lyrics):
        toks = word_tokenize(lyrics.replace('Lyrics', '').replace('You might also like', '').replace('Embed', ''))
        #toks = [tok for tok in toks if tok.isalpha() and tok.lower() not in self.stopwords] # removes punc
        toks = [tok for tok in toks if tok.isalpha()]
        lyrics_embed = []

        for tok in toks:
            vec = self.model.get_word_vector(tok)
            lyrics_embed.append(vec)

        return self.pad(lyrics_embed)

    def pad(self, lyrics_embed):
        lyrics_len = len(lyrics_embed)

        if lyrics_len == self.max_len:
            return lyrics_embed
        elif lyrics_len > self.max_len:
            return lyrics_embed[:self.max_len]
        else:
            return lyrics_embed + [[0 for _ in range(100)] for _ in range(self.max_len - lyrics_len)]
