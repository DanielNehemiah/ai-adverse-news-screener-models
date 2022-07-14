import numpy as np
import pandas as pd
import pickle
import re
from transformers import BertTokenizer, BertForSequenceClassification 
from scipy.special import softmax

from allennlp_improvised_coref import predictor,nlp,coref_resolved_improved

finbert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3) 
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes="(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " "+ text + "  "
    text = text.replace("\n"," ")

    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.]"," \\1<prd>", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]"+ alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>", text) 
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]","\\1<prd>", text) 
    text = re.sub(" "+ alphabets + "[.]","\\1<prd>", text)
    
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text =text.replace("?\"","\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace(";",";<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences= [s.strip() for s in sentences]
    return sentences

def coref_res(articles):
    coref_res_articles = []
    for text in articles:
        prediction = predictor.predict(document=text)
        clusters = prediction['clusters']
        doc = nlp(text)
        text = coref_resolved_improved(doc,clusters)
        coref_res_articles.append(text)
    return coref_res_articles

def spacy_entity_finder(sentence):
    temp = []
    entities = nlp(sentence).ents
    for entity in entities:
        if entity.label_ in ['ORG','PERSON']:
            temp.append({'text': entity.text.strip(),'label': entity.label_})
    org_and_name_entities = []
    [org_and_name_entities.append(x) for x in temp if x not in org_and_name_entities]
    return org_and_name_entities

def news_analyzer(articles):
    # labels = {0:'positive', 1:'negative', 2:'neutral'}
    df = pd.DataFrame({},columns = ['sentences','entities','pos_neg_neu'])
    articles = coref_res(articles)
    for idx,text in enumerate(articles):
        # split into sentences
        sentences = split_into_sentences(text)
        # tokenize sentences
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        # sentiment analysis
        outputs = finbert(**inputs)[0]
        outputs = softmax(outputs.detach().numpy(),axis=1)
        # extract negative sentences
        isneg = lambda x: True if np.argmax(x)==1 else False
        ind = list(map(isneg, outputs))
        sentences = [sentences[i] for i,neg in enumerate(ind) if neg==True]
        outputs = [list(outputs[i]) for i,neg in enumerate(ind) if neg==True]
        # extract ORG and PERSON entities
        entities = [spacy_entity_finder(sentence) for sentence in sentences]
        df.loc[idx] = [sentences, entities, outputs]
    return df