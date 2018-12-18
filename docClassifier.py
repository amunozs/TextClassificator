#!/usr/bin/python
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

from gensim import corpora, models, similarities
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity

sample_glossary = ["deportes", "futbol", "jugar", "política", "presidente", "España", "salud", "medicina", "hospital"]

sports_vector = [[1,1,1,0,0,0,0,0,0]]
politics_vector = [[0,0,0,1,1,1,0,0,0]]
health_vector = [[0,0,0,0,0,0,1,1,1]]

# sports_vector = [(0,1), (1,1), (2,1)]

# politics_vector = [(3,1), (4,1), (5,1)]
# health_vector = [(6,1), (7,1), (8,1)]

sample_corpus = [ "Juego al futbol",
              		"El ganador del partido de ayer, acaba en el hospital",
              		"El presidente de España ",
              		"Nueva medicina en de España ",
              		"Un partido político nuevo ",
              		"Vaya partidazo de futbol",
              		"Dame mis medicinas, quiero mis medicinas"
				]

sample_test = "Vamos a ver si jugamos un partido de futbol con y no acabo en el hospital"

sample_labels = ["deportes", "deportes", "politica", "salud", "politics", "deportes", "salud"]

def preprocess_document(doc):
	stopset = set(stopwords.words('spanish'))
	stemmer = PorterStemmer()
	tokens = wordpunct_tokenize(doc)
	clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
	final = [stemmer.stem(word) for word in clean]
	return final

def create_dictionary(glossary):
	pdocs = [preprocess_document(doc) for doc in glossary]
	dictionary = corpora.Dictionary(pdocs)
	dictionary.save('/tmp/vsm.dict')
	return dictionary

def docs2bows(corpus, dictionary):
	docs = [preprocess_document(d) for d in corpus]
	vectors = [dictionary.doc2bow(doc) for doc in docs]
	corpora.MmCorpus.serialize('/tmp/vsm_docs.mm', vectors)
	return vectors

def create_TF_IDF_model(docs, glossary):
	dictionary = create_dictionary(glossary)
	vectors = docs2bows(docs, dictionary)
	#loaded_corpus = corpora.MmCorpus('/tmp/vsm_docs.mm')
	tfidf = models.TfidfModel(vectors)
	return tfidf, dictionary, vectors
    

# def launch_query(corpus, q):
# 	tfidf, dictionary, vectors = create_TF_IDF_model(corpus)
# 	index = similarities.MatrixSimilarity(vectors, num_features=len(dictionary))
# 	pq = preprocess_document(q)
# 	vq = dictionary.doc2bow(pq)
# 	qtfidf = tfidf[vq]
# 	sim = index[qtfidf]
# 	ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
# 	for doc, score in ranking:
# 		print ("[ Score = " + "%.3f" % round(score,3) + "] " + corpus[doc]); 


def read_glossary():
    return

def label_documents():
    return


def train(corpus, glossary):
    tfidf, dictionary, vectors = create_TF_IDF_model(corpus, glossary)
    print(tfidf)
    print("DICCIONARIO: ", dictionary)
    print("VECTORES: ", vectors)
    index = similarities.MatrixSimilarity(vectors, num_features=len(dictionary))
    pq = preprocess_document(sample_test)
    vq = dictionary.doc2bow(pq)
    print("TEXTO TEST", vq)
    qtfidf = tfidf[vq]
    print("TEXTO TEST TFIDF", qtfidf)
    sim = index[qtfidf]
    
    vq = [[0,1,1,0,0,0,0,0,1]]
    print("GLOSARIO EJEMPLO: ", sample_glossary, "\n")
    print("DOCS TRAIN: ", sample_corpus, "\n")
    print("DOC TEST: ",sample_test , "\n")
    print("SIMILARIDAD CON CADA DOC:", sim, "\n")
    print("SIM CON VEC DEPORTES ", cosine_similarity(vq,sports_vector))
    print("SIM CON VEC POLITICA ", cosine_similarity(vq,politics_vector))
    print("SIM CON VEC SALUD ", cosine_similarity(vq,health_vector))


train(sample_corpus, sample_glossary)

