__author__ = 'Iosu'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

stopwords = ['?l', '?sta', '?stas', '?ste', '?stos', '?ltima', '?ltimas', '?ltimo', '?ltimos', 'a', 'a?adi?', 'a?n',
             'actualmente', 'adelante', 'adem?s', 'afirm?', 'agreg?', 'ah?', 'ahora', 'al', 'alg?n', 'algo', 'alguna',
             'algunas', 'alguno', 'algunos', 'alrededor', 'ambos', 'ante', 'anterior', 'antes', 'apenas',
             'aproximadamente', 'aqu?', 'as?', 'asegur?', 'aunque', 'ayer', 'bajo', 'bien', 'buen', 'buena', 'buenas',
             'bueno', 'buenos', 'c?mo', 'cada', 'casi', 'cerca', 'cierto', 'cinco', 'coment?', 'como', 'con', 'conocer',
             'consider?', 'considera', 'contra', 'cosas', 'creo', 'cual', 'cuales', 'cualquier', 'cuando', 'cuanto',
             'cuatro', 'cuenta', 'da', 'dado', 'dan', 'dar', 'de', 'debe', 'deben', 'debido', 'decir', 'dej?', 'del',
             'dem?s', 'dentro', 'desde', 'despu?s', 'dice', 'dicen', 'dicho', 'dieron', 'diferente', 'diferentes',
             'dijeron', 'dijo', 'dio', 'donde', 'dos', 'durante', 'e', 'ejemplo', 'el', 'ella', 'ellas', 'ello',
             'ellos', 'embargo', 'en', 'encuentra', 'entonces', 'entre', 'era', 'eran', 'es', 'esa', 'esas', 'ese',
             'eso', 'esos', 'est?', 'est?n', 'esta', 'estaba', 'estaban', 'estamos', 'estar', 'estar?', 'estas', 'este',
             'esto', 'estos', 'estoy', 'estuvo', 'ex', 'existe', 'existen', 'explic?', 'expres?', 'fin', 'fue', 'fuera',
             'fueron', 'gran', 'grandes', 'ha', 'hab?a', 'hab?an', 'haber', 'habr?', 'hace', 'hacen', 'hacer',
             'hacerlo', 'hacia', 'haciendo', 'han', 'hasta', 'hay', 'haya', 'he', 'hecho', 'hemos', 'hicieron', 'hizo',
             'hoy', 'hubo', 'igual', 'incluso', 'indic?', 'inform?', 'junto', 'la', 'lado', 'las', 'le', 'les', 'lleg?',
             'lleva', 'llevar', 'lo', 'los', 'luego', 'lugar', 'm?s', 'manera', 'manifest?', 'mayor', 'me', 'mediante',
             'mejor', 'mencion?', 'menos', 'mi', 'mientras', 'misma', 'mismas', 'mismo', 'mismos', 'momento', 'mucha',
             'muchas', 'mucho', 'muchos', 'muy', 'nada', 'nadie', 'ni', 'ning?n', 'ninguna', 'ningunas', 'ninguno',
             'ningunos', 'no', 'nos', 'nosotras', 'nosotros', 'nuestra', 'nuestras', 'nuestro', 'nuestros', 'nueva',
             'nuevas', 'nuevo', 'nuevos', 'nunca', 'o', 'ocho', 'otra', 'otras', 'otro', 'otros', 'para', 'parece',
             'parte', 'partir', 'pasada', 'pasado', 'pero', 'pesar', 'poca', 'pocas', 'poco', 'pocos', 'podemos',
             'podr?', 'podr?n', 'podr?a', 'podr?an', 'poner', 'por', 'porque', 'posible', 'pr?ximo', 'pr?ximos',
             'primer', 'primera', 'primero', 'primeros', 'principalmente', 'propia', 'propias', 'propio', 'propios',
             'pudo', 'pueda', 'puede', 'pueden', 'pues', 'qu?', 'que', 'qued?', 'queremos', 'qui?n', 'quien', 'quienes',
             'quiere', 'realiz?', 'realizado', 'realizar', 'respecto', 's?', 's?lo', 'se', 'se?al?', 'sea', 'sean',
             'seg?n', 'segunda', 'segundo', 'seis', 'ser', 'ser?', 'ser?n', 'ser?a', 'si', 'sido', 'siempre', 'siendo',
             'siete', 'sigue', 'siguiente', 'sin', 'sino', 'sobre', 'sola', 'solamente', 'solas', 'solo', 'solos',
             'son', 'su', 'sus', 'tal', 'tambi?n', 'tampoco', 'tan', 'tanto', 'ten?a', 'tendr?', 'tendr?n', 'tenemos',
             'tener', 'tenga', 'tengo', 'tenido', 'tercera', 'tiene', 'tienen', 'toda', 'todas', 'todav?a', 'todo',
             'todos', 'total', 'tras', 'trata', 'trav?s', 'tres', 'tuvo', 'un', 'una', 'unas', 'uno', 'unos', 'usted',
             'va', 'vamos', 'van', 'varias', 'varios', 'veces', 'ver', 'vez', 'y', 'ya', 'yo']


def bow(list_of_words, vec='Unknown'):
    # print list_of_words
    print "Creating the bag of words...\n"

    #
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.

    if (vec.__eq__('tfidf')):
        vectorizer = TfidfVectorizer(analyzer="char", \
                                     ngram_range=[3, 4], \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=stopwords, \
                                     max_features=5000)
    else:
        vectorizer = CountVectorizer(analyzer="char", \
                                     ngram_range=[3, 4], \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=stopwords, \
                                     max_features=5000)


    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.


    train_data_features = vectorizer.fit_transform(list_of_words)

    # Numpy arrays are easy to work with, so convert the result to an
    # array


    train_data_features = train_data_features.toarray()


    # print train_data_features.shape


    vocab = vectorizer.get_feature_names()
    # print vocab


    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # print dist
    # For each, print the vocabulary word and the number of times it
    # appears in the training set

    dictionary = []
    for tag, count in zip(vocab, dist):
        dictionary.append((tag, count))
        # print count, tag

    dictionary = sorted(dictionary, key=lambda x: x[1])

    return dictionary, train_data_features, vectorizer