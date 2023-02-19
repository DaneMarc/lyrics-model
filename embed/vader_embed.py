from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Vader:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def lyrics_to_vec(self, lyrics):
        lyrics = lyrics.replace('Lyrics', '').replace('You might also like', '').replace('Embed', '')
        vec = self.vader.polarity_scores(lyrics)
        vec = list(vec.values())
        return vec
