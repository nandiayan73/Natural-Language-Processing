import spacy
nlp=spacy.load('en_core_web_sm')

# Showing all the entities present in the text:
def show_ents(doc):
    if(doc.ents):
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print("No named entity found.")
doc=nlp(u"Ratan Tata is the owner of the company tata!")

# Adding a new entity
from spacy.tokens import Span

ORG=doc.vocab.strings[u"ORG"]

new_entity=Span(doc,8,9,label=ORG)
doc.ents=list(doc.ents)+[new_entity]

# show_ents(doc)
doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
          u'If successful, the vacuum-cleaner will be our first product.')
from spacy.matcher import Matcher
matcher=Matcher(nlp.vocab)
phrase_list=['vacuum cleaner','vacuum-cleaner']


phrase_pattenrs=[nlp(text) for text in phrase_list]


matcher.add('newproduct',*phrase_pattenrs)


matches=matcher(doc)

print(matches)

for match in matches:
    print(doc[match[1]:match[2]])

PROD=doc.vocab.strings[u"PRODUCT"]


new_ents=[Span(doc,match[1],match[2],label=PROD) for match in matches]

doc.ents=list(doc.ents)+new_ents
show_ents(doc)

