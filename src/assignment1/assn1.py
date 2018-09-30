import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re
import string
from scipy.cluster.hierarchy import linkage,dendrogram
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import codecs

#nltk.download()

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def tokenize_stem(raw):
    
    """
    lower casing the string
    """
    raw = raw.lower()
    
    
    """
    removing numbers
    """
    translator = str.maketrans('', '', string.digits)
    text_ = raw.translate(translator)
    
    """
    removing punctuations from text
    """
    translator1 = str.maketrans('', '', string.punctuation)
    text = text_.translate(translator1)
    
    #print("raw {} , text_ {} and text{}".format(len(raw),len(text_),len(text)))
    """
    tokenize the text
    """
    words =[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    """
    remove stop words
    """
    stop_words = set(stopwords.words("english"))
    #print(len(stop_words))
    stop_words.add("search")
    stop_words.add("engine")
    #print(len(stop_words))
    filtered_words = [w for w in words if w not in stop_words]
    
    """
    used for removing terms made of numbers and punctuations
    filtered_words_1 = []
    for token in filtered_words:
        if re.search('[a-zA-Z]', token):
            filtered_words_1.append(token)
    """
    #print("tokens",len(filtered_words))
    
    """
    perform stemming on text
    """
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    return stemmed_words

def get_jacc_distance_matrix(st):
    c= []
    f = False
    for item in st:
        for t in st:
            c.append(1-jaccard_similarity(item,t))
        if not f:
            res = [i for i in c]
            f = True
        else:
            res = np.vstack((res,c))
        c.clear()
    return res

def get_jacc_similarity_matrix(st):
    c= []
    f = False
    for item in st:
        for t in st:
            c.append(jaccard_similarity(item,t))
        if not f:
            res = [i for i in c]
            f = True
        else:
            res = np.vstack((res,c))
        c.clear()
    return res

def get_text(filename):
    """
    f= open(filename,"rb")
    raw = f.read()
    """
    with codecs.open(filename, "r",encoding='utf-8', errors='ignore') as fdata:
        raw = fdata.read() 
    return raw 

"""
def cluster_texts_kmeans(texts, clusters=3):
    # Transform texts to Tf-Idf coordinates and cluster texts using K-Means 
    vectorizer = TfidfVectorizer(tokenizer=tokenize_stem,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
    print(km_model.labels_)
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering
"""


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def main():
    s = "msc-plagiarism-assigment/ass1-"
    l = ["1349","422","734","808","936","1019","1037","1046","1138","1147","202","211","321","440","505","532","541","606","743","817","826","909"]
    filenames = [s+item+".txt" for item in l]
    #print(len(filenames))
    texts = []
    for names in filenames:
        texts.append(get_text(names))
    
    """
    #perform K-means clustering
    
    cluster = cluster_texts_kmeans(texts,11)
    print("\nresult of k-mean clustering")
    for key in cluster.keys():
        files = []
        for item in cluster[key]:
            files.append(filenames[int(item)])
        print("cluster {}: {}".format(key,files))    
   """
    
    """
    jaccard similarity matrix
    """
    jacc_sm = get_jacc_similarity_matrix(texts)
    print("\nsimilarity matrix:\n",jacc_sm)
    
    """
    jaccard dissimilarity matrix
    """
    res = get_jacc_distance_matrix(texts)
    print("\n\ndistance matrix:\n",res)
    
    """
    making condensed distance matrix of shape nC2
    """
    distArray = ssd.squareform(res)
    #print(distArray)
    
    """
    making dendogram
    """
    Z = linkage(distArray)
    fig = plt.figure(figsize=(25, 10))
    
    fancy_dendrogram(
    Z,
    labels = l,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    max_d=.1
)
    #dn = dendrogram(Z,labels=l)
    plt.show()
    
if __name__ == "__main__":
    main()
