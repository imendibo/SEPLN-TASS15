__author__ = 'Iosu'


import xmlreader as xml

if __name__ == "__main__":

    xmlTrainFile = '../DATA/general-tweets-train-tagged.xml'
    tweets = xml.readXML(xmlTrainFile)

    for tweet in tweets:
        preprocess(tweet.content)