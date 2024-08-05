import spacy

nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(u'John Cena is looking for buying niggas. are you good?')
doc2=nlp(u" john is looking for a funny puppy.")
doc3=nlp(u"JOHN IS gay,handsome,cute and dashing ok.")

for token in doc.sents:
    print(token)

# print(doc[7]) 
# print(doc[7].is_sent_end)
for chunk in doc3.noun_chunks:
    print(chunk.text)