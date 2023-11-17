import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

def senti_test(texts):
    analyzer = SentimentIntensityAnalyzer()
    for text in texts:
        scores = analyzer.polarity_scores(text)
        # print(text)
        # print(scores)   
    
    print("Overall lyrics is : ", scores)
    print("Lyrics was rated as ", scores['neg']*100, "% Negative")
    print("Lyrics was rated as ", scores['neu']*100, "% Neutral")
    print("Lyrics was rated as ", scores['pos']*100, "% Positive")
 
    print("Lyrics Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if scores['compound'] >= 0.05 :
        print("Positive")
 
    elif scores['compound'] <= - 0.05 :
        print("Negative")
 
    else :
        print("Neutral")
test = all_lyrics['Loski_lyrics.csv']
print(test)
# print(senti_test(all_lyrics))
