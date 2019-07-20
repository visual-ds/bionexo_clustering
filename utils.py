
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    return txt

def get_clean_list_docs(nlp, df, col_name):
    doc = (re.sub("[^a-zA-ZÀ-ú']+", ' ', str(row)).lower() for row in df[col_name])
    txt = [cleaning(doc) for doc in nlp.pipe(doc, batch_size=5000, n_threads=-1)]
    return txt

def doc_to_sum_vector(str1, embed_map, dim):
    try:
        sentenceA = str1.split()
        vecA = np.zeros(dim)
        count = 0
        for word in sentenceA:
            if word in embed_map:
                count += 1
                vecA += embed_map[word]

        vecA = vecA/max(count, 1)
    except:
        return np.zeros(dim)

    return vecA


def get_embe_sim(str1, str2, embed_map, dim):
    
    #print(str1, '\n', str2)

    sentenceA = str1.split()
    sentenceB = str2.split()
    
    #print(sentenceA, '\n', sentenceB)

    vecA = np.zeros(dim)
    vecB = np.zeros(dim)

    count = 0
    for word in sentenceA:
        if word in embed_map:
            count += 1
            vecA += embed_map[word]

    vecA = vecA/max(count, 1)

    count = 0
    for word in sentenceB:
        if word in embed_map:
            vecB += embed_map[word]
            count += 1

    vecB = vecB/max(count, 1)
    
    #print(vecA, '\n', vecB)

    similarity = cosine_similarity([vecA], [vecB])
    similarity = similarity[0][0]
    return similarity


def get_embe_sim_vec(vecA, vecB, embed_map, dim):
    similarity = cosine_similarity([vecA], [vecB])
    similarity = similarity[0][0]
    return similarity