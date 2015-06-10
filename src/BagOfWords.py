__author__ = 'Iosu'

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



stopwords = set(
    ['un', 'una', 'unas', 'unos', 'uno', 'sobre', 'todo', 'tambi?n', 'tras', 'otro', 'alg?n', 'alguno', 'alguna',
     'algunos', 'algunas', 'ser', 'es', 'soy', 'eres', 'somos', 'sois', 'estoy', 'esta', 'estamos', 'estais', 'estan',
     'como', 'en', 'para', 'atras', 'porque', 'por qu?', 'estado', 'estaba', 'ante', 'antes', 'siendo', 'ambos', 'pero',
     'por', 'poder', 'puede', 'puedo', 'podemos', 'podeis', 'pueden', 'fui', 'fue', 'fuimos', 'fueron', 'hacer', 'hago',
     'hace', 'hacemos', 'haceis', 'hacen', 'cada', 'fin', 'incluso', 'primero', 'desde', 'conseguir', 'consigo',
     'consigue', 'consigues', 'conseguimos', 'consiguen', 'ir', 'voy', 'va', 'vamos', 'vais', 'van', 'vaya', 'gueno',
     'ha', 'tener', 'tengo', 'tiene', 'tenemos', 'teneis', 'tienen', 'el', 'la', 'lo', 'las', 'los', 'su', 'aqui',
     'mio', 'tuyo', 'ellos', 'ellas', 'nos', 'nosotros', 'vosotros', 'vosotras', 'si', 'dentro', 'solo', 'solamente',
     'saber', 'sabes', 'sabe', 'sabemos', 'sabeis', 'saben', 'ultimo', 'largo', 'bastante', 'haces', 'muchos',
     'aquellos', 'aquellas', 'sus', 'entonces', 'tiempo', 'verdad', 'verdadero', 'verdadera', 'cierto', 'ciertos',
     'cierta', 'ciertas', 'intentar', 'intento', 'intenta', 'intentas', 'intentamos', 'intentais', 'intentan', 'dos',
     'bajo', 'arriba', 'encima', 'usar', 'uso', 'usas', 'usa', 'usamos', 'usais', 'usan', 'emplear', 'empleo',
     'empleas', 'emplean', 'ampleamos', 'empleais', 'valor', 'muy', 'era', 'eras', 'eramos', 'eran', 'modo', 'bien',
     'cual', 'cuando', 'donde', 'mientras', 'quien', 'con', 'entre', 'sin', 'trabajo', 'trabajar', 'trabajas',
     'trabaja', 'trabajamos', 'trabajais', 'trabajan', 'podria', 'podrias', 'podriamos', 'podrian', 'podriais', 'yo',
     'aquel'])

def bow(list_of_words):
    # print list_of_words
    print "Creating the bag of words...\n"
    # from sklearn.feature_extraction.text import CountVectorizer
    #
    # # Initialize the "CountVectorizer" object, which is scikit-learn's
    # # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = stopwords,   \
                                 max_features = 5000)

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