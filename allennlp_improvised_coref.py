# from allennlp.predictors.predictor import Predictor
# model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
# predictor = Predictor.from_path(model_url)
# with open('allennlp_coref.pickle', 'wb') as handle:
#     pickle.dump(predictor, handle, protocol=pickle.HIGHEST_PROTOCOL)

import pickle
import en_core_web_sm
nlp = en_core_web_sm.load()

coref_model = "allennlp_coref.pickle"
with open(coref_model,'rb') as handle:
    predictor = pickle.load(handle)

def get_span_noun_indices(doc, cluster):
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices

def get_cluster_head(doc, cluster,noun_indices):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start:head_end+1]
    return head_span, [head_start, head_end]

def is_containing_other_spans(span, all_spans):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

def coref_resolved_improved(doc, clusters):
    resolved = [tok.text_with_ws for tok in doc]
    all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans
    for cluster in clusters:
        noun_indices = get_span_noun_indices(doc, cluster)
        if noun_indices:
            mention_span, mention = get_cluster_head(doc, cluster, noun_indices)
            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    final_token = doc[coref[1]]
                    if final_token.tag_ in ["PRP$", "POS"]:
                        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
                    else:
                        resolved[coref[0]] = mention_span.text + final_token.whitespace_

                    for i in range(coref[0] + 1, coref[1] + 1):
                        resolved[i] = ""
    return "".join(resolved)

'''
from allennlp_improvised_coref import *
text = "He is a great actor!, he said about John Travolta. After saying this Ram ran away."
prediction = predictor.predict(document=text)  # get prediction
clusters = prediction['clusters']
doc = nlp(text)
coref_resolved_improved(doc,clusters)
>>> 'John Travolta is a great actor!, Ram said about John Travolta. After saying this Ram ran away.'
'''