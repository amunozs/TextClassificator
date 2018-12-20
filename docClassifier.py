#!/usr/bin/python
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

from gensim import corpora, models, similarities
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
import csv
import glob

#sample_glossary = ["deportes", "futbol", "jugar", "política", "presidente", "España", "tecnologia", "medicina", "hospital"]

sports_vector = [[1,1,1,0,0,0,0,0,0]]
politics_vector = [[0,0,0,1,1,1,0,0,0]]
health_vector = [[0,0,0,0,0,0,1,1,1]]

# sports_vector = [(0,1), (1,1), (2,1)]

# politics_vector = [(3,1), (4,1), (5,1)]
# health_vector = [(6,1), (7,1), (8,1)]

sample_test_deportes = 'Marcelo compareció en sala de prensa en la previa del duelo entre el Real Madrid y el Kashima Antlers, la segunda semifinal del Mundial de Clubes 2018 que se disputa en Abu Dabi.Mérito del Madrid: "Creo que el mérito es la unión que tenemos dentro del vestuario, el equipo lleva jugando mucho tiempo junto. Pero no se para aquí, tenemos hambre de ganar títulos, estamos en una competición corta".Nuevo reto: "No pienso en el individual, está por encima el colectivo, la unión. Somos una familia, muy amigos dentro y fuera del campo y se nota. Disputamos una competición muy corta pero que vale mucho". Mourinho: "Estoy aquí para hablar del Mundial de Clubes. Es una pena porque es un gran entrenador y ahora está sin club. No soy yo quien decide si vuelve pero le agradezco lo que hizo por mí en el Madrid".Situación de los jóvenes: "Cuando llegué al Madrid era muy joven y los que tenían experiencia me ayudaron a mí. Y nosotros con la edad que tenemos intentamos ayudar a los jóvenes. Ellos tienen que estar a gusto, trabajar y hacerlo todo para dejar al Madrid donde tiene que estar que es arriba".Habló de Isco: "Todos sabemos de la calidad, quizás el que más calidad que tiene. Se le esta dando mucha importancia a algo que no tiene mucho sentido. Algunos juegan más y otros menos. Los jugadores tenemos momentos. No hay que darle más vuelta. Está en el Real Madrid y no hace recuperarle".'



def preprocess_document(doc):
	stopset = set(stopwords.words('spanish'))
	stemmer = SnowballStemmer('spanish')
	tokens = wordpunct_tokenize(doc)
	clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
	final = [stemmer.stem(word) for word in clean]
	return final

def create_dictionary(glossary):
	pdocs = [preprocess_document(doc) for doc in glossary]
	dictionary = corpora.Dictionary(pdocs)
	#dictionary.save('/tmp/vsm.dict')
	return dictionary

def docs2bows(corpus, dictionary):
	docs = [preprocess_document(d) for d in corpus]
	vectors = [dictionary.doc2bow(doc) for doc in docs]
	#corpora.MmCorpus.serialize('/tmp/vsm_docs.mm', vectors)
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


def read_glossary(dir):
	fns = glob.glob(dir+'*.csv')
	glossary = []
	for fn in fns:
		with open(fn, 'r', encoding="utf8") as csvfile:
			reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			glossary += [row[0] for row in reader]
	return glossary

def read_texts(dir):
	fns = glob.glob(dir+'*.txt')
	corpus = []
	for fn in fns:
		with open(fn, 'r', encoding="utf8") as txtfile:
			content = txtfile.read()
			corpus.append(content)
	return corpus

def label_documents():
    return


def train(corpus, glossary):
    tfidf, dictionary, vectors = create_TF_IDF_model(corpus, glossary)
    #print(tfidf)
    #print("DICCIONARIO: ", dictionary)
    #print("VECTORES: ", vectors)
    index_sim = similarities.MatrixSimilarity(tfidf[vectors], num_features=len(dictionary))
    return dictionary, tfidf, index_sim
    
    #vq = [[0,1,1,0,0,0,0,0,1]]
    #print("GLOSARIO EJEMPLO: ", glossary, "\n")
    #print("DOCS TRAIN: ", sample_corpus, "\n")
    #print("DOC TEST: ",sample_test_deportes , "\n")
    #print("SIMILARIDAD CON CADA DOC:", sim, "\n")
    #print("SIM CON VEC DEPORTES ", cosine_similarity(vq,sports_vector))
    #print("SIM CON VEC POLITICA ", cosine_similarity(vq,politics_vector))
    #print("SIM CON VEC tecnologia ", cosine_similarity(vq,health_vector))

def test(dictionary, tfidf, index_sim, test_docs, doc_index):
	acc = 0

	for doc in test_docs:
		pdoc = preprocess_document(doc)
		vdoc = dictionary.doc2bow(pdoc)
		doc_tfidf = tfidf[vdoc]
		sim = index_sim[doc_tfidf]
		ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
		count = 0
		K = 1
		for doc, score in ranking[:K]:
			if doc < doc_index and doc >= doc_index - 170:
				count += 1

		if count >= 1:
			acc +=1
			#print (doc ," [ Score = " + "%.3f" % round(score,3) + "] " , corpus[doc])

	return acc*100/len(test_docs)

glossary = read_glossary('./Glosario/')

corpus_deportes = read_texts('./Deportes/')
corpus_politica = read_texts('./Politica/')
corpus_tecnologia = read_texts('./Tecnologia/')
corpus = corpus_deportes + corpus_politica + corpus_tecnologia

test_deportes = read_texts('./Test_Deportes/')
test_politica = read_texts('./Test_Politica/')
test_tecnologia = read_texts('./Test_Tecnologia/')

dictionary, tfidf, index_sim = train(corpus, glossary)

acc_dep = test(dictionary, tfidf, index_sim, test_deportes, 170)
print("Precisión Deportes: ", acc_dep)

acc_pol = test(dictionary, tfidf, index_sim, test_politica, 170*2)
print("Precisión Política: ", acc_pol)

acc_tec = test(dictionary, tfidf, index_sim, test_tecnologia, 170*3)
print("Precisión Tecnología: ", acc_tec)