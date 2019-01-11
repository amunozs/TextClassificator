#!/usr/bin/python
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import warnings
import os
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

from gensim import corpora, models, similarities
from operator import itemgetter
import csv
import glob

#SEP = "\\" # windows
SEP = "/" # unix
categories = [0,1,2]
cat_string = ["Deportes","Política","Tecnología"]
cat_train_size = 170

# Crea el modelo TF-IDF a partir de un conjunto de documentos y su glosario
def create_TF_IDF_model(corpus, glossary):
	dictionary = create_dictionary(glossary)
	vectors = docs2bows(corpus, dictionary)
	tfidf = models.TfidfModel(vectors)
	return tfidf, dictionary, vectors

# Crea el diccionario a partir de un glosario de entrada
def create_dictionary(glossary):
	pdocs = [preprocess_document(doc) for doc in glossary]
	dictionary = corpora.Dictionary(pdocs)
	return dictionary

# Preprocesa un texto: elimina stepwords, saca la raiz, tokeniza, y quita tokens cortos
def preprocess_document(doc):
	stopset = set(stopwords.words('spanish'))
	stemmer = SnowballStemmer('spanish')
	tokens = wordpunct_tokenize(doc)
	clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
	final = [stemmer.stem(word) for word in clean]
	return final

# Convierte un corpus de textos a vectores de tipo Bag of Words según el diccionario
def docs2bows(corpus, dictionary):
	docs = [preprocess_document(d) for d in corpus]
	vectors = [dictionary.doc2bow(doc) for doc in docs]
	return vectors
    
# Crea un glosario leyendo los archivos de un directorio
def read_glossary(dir):
	fns = glob.glob(dir+'*.csv')
	glossary = []
	for fn in fns:
		with open(fn, 'r', encoding="utf8") as csvfile:
			reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
			glossary += [row[0] for row in reader]
	return glossary

# Lee un corpus de documentos de un directorio
def read_texts(dir):
	fns = glob.glob(dir+'*.txt')
	corpus = []
	for fn in fns:
		with open(fn, 'r', encoding="utf8") as txtfile:
			content = txtfile.read()
			corpus.append(content)
	return corpus

# Entrada: un corpus de textos de entrenamiento y su glosario
# Crea su diccionario, el modelo TF-IDF y la matriz de similaridad de coseno
def train(corpus, glossary):
    tfidf, dictionary, vectors = create_TF_IDF_model(corpus, glossary)
    index_sim = similarities.MatrixSimilarity(tfidf[vectors], num_features=len(dictionary))
    return dictionary, tfidf, index_sim
	

# Test con KNN
def testKNN(dictionary, tfidf, index_sim, test_docs):
	tp = [0,0,0]
	fp = [0,0,0]
	fn = [0,0,0]
	for cat in categories:
		for doc in test_docs[cat]:
			pdoc = preprocess_document(doc)
			vdoc = dictionary.doc2bow(pdoc)
			doc_tfidf = tfidf[vdoc]
			sim = index_sim[doc_tfidf]
			ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
			# 1-NN
			cdoc = ranking[:1][0][0]
			category = int(cdoc/cat_train_size)
			if (category == cat):
				tp[category]+=1
			else:
				fn[category]+=1
				fp[cat]+=1


	return evaluate(tp, fp, fn)

def test(dictionary, tfidf, index_sim, test_docs, folders):
	tp = [0,0,0]
	fp = [0,0,0]
	fn = [0,0,0]
	for category in categories:
		cdocs = classify(dictionary, tfidf, index_sim, test_docs[category],folders)
		for cdoc in cdocs:
			if (cdoc == category):
				tp[category]+=1
			else:
				fn[category]+=1
				fp[cdoc]+=1

	return evaluate(tp, fp, fn)

# Clasifica los textos en las 3 categorías, introduciéndolos en sus carpetas correspondientes
def classify(dictionary, tfidf, index_sim, test_docs, folders):
	cdocs = []
	for doc in test_docs:
		rankdoc = classifyDoc(doc,dictionary, tfidf, index_sim)
		category = rankdoc[0]
		cdocs.append(category)
		filename = folders[category] + SEP + str(rankdoc[1]) + ".txt" 
		if not os.path.exists(folders[category]):
			os.makedirs(folders[category])
		f = open(filename,"wb+") 
		f.write(doc.encode("utf-8"))
		f.close()
	return cdocs

# Clasifica un texto a una categoría según las similaridad a la media de la categoría
# Y devuelve dicha similaridad
# Deportes: 0
# Política: 1
# Tecnología: 2
def classifyDoc(doc, dictionary, tfidf, index_sim):
	cat_train_size = 170
	pdoc = preprocess_document(doc)
	vdoc = dictionary.doc2bow(pdoc)
	doc_tfidf = tfidf[vdoc]
	sim = index_sim[doc_tfidf]
		
	cdoc = []
	cdoc.append(0)
	category = 0
	for doc, score in enumerate(sim):
		if doc - category * cat_train_size >= cat_train_size:
			category += 1
			cdoc.append(0)
		cdoc[category] += score/cat_train_size

	ranking = sorted(enumerate(cdoc), key=itemgetter(1), reverse=True)
	return ranking[0]	

# Calcula Precisión y Exhaustividad
def evaluate(tp, fp, fn):
	precision = 0
	recall = 0
	for category in categories:
		prec = tp[category]/(tp[category]+fp[category])
		rec = tp[category]/(tp[category]+fn[category])
		print("Precision ", cat_string[category], ": ", prec)
		print("Recall ", cat_string[category], ": ", rec)
		precision += prec
		recall += rec
		
	return precision/3, recall/3


def main():
	glossary = read_glossary('./Glossary/')

	corpus_deportes = read_texts('./Train/Deportes/')
	corpus_politica = read_texts('./Train/Politica/')
	corpus_tecnologia = read_texts('./Train/Tecnologia/')
	corpus = corpus_deportes + corpus_politica + corpus_tecnologia

	test_deportes = read_texts('./Test/Deportes/')
	test_politica = read_texts('./Test/Politica/')
	test_tecnologia = read_texts('./Test/Tecnologia/')
	test_docs = [test_deportes, test_politica, test_tecnologia]

	dictionary, tfidf, index_sim = train(corpus, glossary)

	result_folders = ["Results"+SEP+"Deportes", "Results"+SEP+"Politica", "Results"+SEP+"Tecnologia"]

	print ("\n KNN")
	precisionKnn, recallKnn = testKNN(dictionary,tfidf, index_sim, test_docs)
	print ("Total precision: ", precisionKnn)
	print ("Total recall: ", recallKnn)

	print ("\n Centroid-based")
	precision, recall = test(dictionary, tfidf, index_sim, test_docs, result_folders)
	print ("Total precision: ", precision)
	print ("Total recall: ", recall)


if __name__ == '__main__':
	main()
	

#precision = tp/(tp+fp)
#recall = tp/(tp+fn)
