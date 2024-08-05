
# In natural language processing (NLP), stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base, or root form. A stemmer is a tool or algorithm used to perform stemming. The goal of stemming is to reduce words to a 
# common form so that different grammatical variations of a word can be 
# treated as the same word, simplifying text analysis.

from nltk.stem.porter import PorterStemmer

p_stemmer=PorterStemmer()

words=['run','runner','ran','fairly','fairness']
for word in words:
    print(word+'----->'+p_stemmer.stem(word))



# words=["eat","eating","ate","eater"]
words=['jumps',"jumping","jump","jumped"]
for word in words:
    print(word+'----->'+p_stemmer.stem(word))
