import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re
import string
from scipy.cluster.hierarchy import linkage,dendrogram
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
import codecs
from pandas import *
import seaborn as sns


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
    translator1 = str.maketrans('', '', string.punctuation+"’"+"�"+"–"+"‘"+"“"+"”"+"●")
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
    with codecs.open(filename, "r",encoding='utf-8', errors='ignore') as fdata:
        raw = fdata.read() 
    return raw

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
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
    s = "Data/ass1-"
    l =["1349","422","734","808","936","1019","1037","1046","1138","1147","202","211","321","440","505","532","541","606","743","817","826","909"]
    filenames = [s+item+".txt" for item in l]
    #print(len(filenames))
    texts = []
    for names in filenames:
        texts.append(tokenize_stem(get_text(names)))
        
    """
    jaccard similarity matrix
    """
    jacc_sm = get_jacc_similarity_matrix(texts)
 
    print(np.min(jacc_sm))
    print(type(jacc_sm))
    #print(jacc_sm)
    #print(min1)
    df = DataFrame(jacc_sm,columns=l,index=l)
    print(df.describe())
    #print("\nSimilarity matrix:\n",DataFrame(jacc_sm))

    plt.title('Similarity Matrix - Visualisation')
    ax = sns.heatmap(df,annot=True,annot_kws={"size":7},cmap="YlGnBu")
    ax.set(xlabel="Documents",ylabel="Documents")
    plt.show()
    
    
    
    """
    jaccard dissimilarity matrix
    """
    res = get_jacc_distance_matrix(texts)
    print("\n\nDistance matrix:\n",DataFrame(res))
    
    """
    making condensed distance matrix of shape nC2
    """
    distArray = ssd.squareform(res)
    #print('Distance matrix', distArray)
    
    """
    making dendogram
    """
    Z = linkage(distArray,method='single')
    print(Z)
    fig = plt.figure(figsize=(20, 10))
    
    fancy_dendrogram(
    Z,
    labels = l,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    max_d=.75
)
    plt.show()
    
    #elbow method
    last = Z[:,2]
    last_rev = last[::-1]
    idxs = np.arange(1,len(last)+1)
    acceleration = np.diff(last,2)
    acceleration_rev  = acceleration[::-1]
    plt.title("Elbow Method - Estimation of number of clusters")
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distance growth (Acceleration)')
    plt.plot(idxs[:-2]+1,acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax()+2
    print("clusters",k)
    
if __name__ == "__main__":
    main()
