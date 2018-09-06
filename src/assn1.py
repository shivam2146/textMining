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

#nltk.download()

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def tokenize_stem(raw):
    
    """
    removing punctuations from text
    """
    translator = str.maketrans('', '', string.punctuation)
    text = raw.translate(translator)
    
    """
    tokenize the text
    """
    words =[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    """
    remove stop words
    """
    stop_words = set(stopwords.words("english"))
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
    f= open(filename)
    raw = f.read()
    return raw 

def cluster_texts_kmeans(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
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

def main():
    filenames = ["History of web search engines.txt","ass1_22.txt"]
    texts = []
    for names in filenames:
        texts.append(get_text(names))
    filenames.extend(["History of web search engines.txt","ass1_22.txt"])
    texts.append(texts[0])
    texts.append(texts[1])
    
    """
    perform K-means clustering
    """
    cluster = cluster_texts_kmeans(texts)
    print("\nresult of k-mean clustering")
    for key in cluster.keys():
        files = []
        for item in cluster[key]:
            files.append(filenames[int(item)])
        print("cluster {}: {}".format(key,files))    
    
    stemmed_words = tokenize_stem(texts[0])
    st1 = tokenize_stem(texts[1])
    #generate data for testing
    st2 = []
    st3 = []
    st4 = []
    for w in stemmed_words:
        st2.append(w)
        st3.append(w)
        st4.append(w)
    st2.extend(["hcdsfellp","fgfwfewfvg","vfvgbgtbvthv","vbrbrtbtvtgvgt"])
    st3.extend(["hevevfergrllp","gfvgtgergg","vtgegergerbvthv"])
    st4.extend(["hellp","gfvg","vtbvthv","vtvtgvgt","revreve","ververv","vreverv"])
    #st = np.stack((stemmed_words,stemmed_words,stemmed_words,st2,stemmed_words,stemmed_words,st2)) #this doesn't work if lengths of lists are different
    st = [stemmed_words,st1,st2,st3,st4]
    #d = pd.DataFrame(st)
    
    """
    jaccard similarity matrix
    """
    jacc_sm = get_jacc_similarity_matrix(st)
    print("\nsimilarity matrix:\n",jacc_sm)
    
    """
    jaccard dissimilarity matrix
    """
    res = get_jacc_distance_matrix(st)
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
    dn = dendrogram(Z)
    plt.show()
    
if __name__ == "__main__":
    main()
