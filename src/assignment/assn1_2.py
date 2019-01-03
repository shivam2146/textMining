from assn1a import get_text
from assn1a import tokenize_stem
from assn1a import fancy_dendrogram
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import linkage,dendrogram
from matplotlib import pyplot as plt
import scipy.spatial.distance as ssd
from pandas import *
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import string
import codecs
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

'''
Computing cosine similarity (Similarity matrix)
'''
def get_cos_similarity_matrix(st):
    c = []
    f = False
    for item in st:
        for t in st:
            c.append(cosine_similarity(item,t)[0][0])
            #print(type(cosine_similarity(item,t)[0][0]))
        if not f:
            res = [i for i in c]
            f = True
        else:
            res = np.vstack((res,c))
        c.clear()
    return res

'''
Computing distance matrix
'''
def get_distance_matrix(st):
    c = []
    f = False
    for item in st:
        for t in st:
            c.append(1-(cosine_similarity(item,t)[0][0]))
        if not f:
            res = [i for i in c]
            f = True
        else:
            res = np.vstack((res,c))
        c.clear()
    return res

'''
K means Clustering function
'''
def cluster_texts_kmeans(texts, clusters=3):
    # Transform texts to Tf-Idf coordinates and cluster texts using K-Means 
    vectorizer = TfidfVectorizer(tokenizer=tokenize_stem,
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True,sublinear_tf=True,smooth_idf=False)
 
    tfidf_model = vectorizer.fit_transform(texts)
    print("tf idf",tfidf_model.shape)
    #print("features",vectorizer.get_feature_names())
    npm_tfidf = tfidf_model.todense()
    sp = tfidf_model.toarray()
    print("sparse",tfidf_model.toarray().shape)
    vocabulary = vectorizer.vocabulary_
    #print(vocabulary)
    sse = {}
    distortions = []
    
    sil_coeff = []
    K = range(2,22)
    for k in K:
        km_model = KMeans(n_clusters=k,n_init=50,max_iter=100)
        km_model.fit(npm_tfidf)
        label = km_model.labels_
        #print(label)
        sil_coeff.append(silhouette_score(npm_tfidf, label, metric='euclidean'))
        #print(km_model.labels_)
        sil_coeff1 = silhouette_score(npm_tfidf, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff1))
        distortions.append(sum(np.min(cdist(npm_tfidf, km_model.cluster_centers_, 'euclidean'), axis=1)) / npm_tfidf.shape[0])
        sse[k] = km_model.inertia_
    print(sse)
    m = 0
    index = -1
    for i in range(len(sil_coeff)):
        if sil_coeff[i] > m :
            index = i
            m = sil_coeff[i]
    print("max {} and at iter {}".format(m,index))
    #Elbow curve - K means
    
    plt.title("Elbow Method - Estimation of number of clusters")
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE error')
    plt.plot(K, distortions,'bx-')
    plt.show()
    plt.title("Elbow Method - Estimation of number of clusters")
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE 2 error')
    plt.plot(list(sse.keys()), list(sse.values()),'bx-')
    plt.show()

    
    km_model = KMeans(n_clusters=index+2)
    km_model.fit(tfidf_model)
    #print(km_model.labels_)    
    
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return npm_tfidf, clustering

def main():
    s = "msc-plagiarism-assigment/ass1-"
    l = ["1349","422","734","808","936","1019","1037","1046","1138","1147","202","211","321","440","505","532","541","606","743","817","826","909"]
    filenames = [s+item+".txt" for item in l]
    #print(len(filenames))
    texts = []
    for names in filenames:
        texts.append(get_text(names))
    
    '''
    Perform K-means clustering
    '''
    npm_tfidf, cluster = cluster_texts_kmeans(texts)
    print("npm tfidf",npm_tfidf.shape)
    
    for i in range(22):
        l = 0
        #print(npm_tfidf[i])
        for j in range(577):
            if(npm_tfidf[i][0][j] != 0):
                l = l+1
        print("i = {}, l={}".format(i,l))
    print("\nResult of k-mean clustering")
    for key in cluster.keys():
        files = []
        for item in cluster[key]:
            files.append(filenames[int(item)])
        print("cluster {}: {}".format(key,files))
    
    """
    Cosine similarity matrix
    """
    cos_sm = get_cos_similarity_matrix(npm_tfidf)
    #print("\nsimilarity matrix:\n",cos_sm)
    
    
    """
    cosine dissimilarity matrix
    """
    dist = get_distance_matrix(npm_tfidf)
    #print("\n\ndistance matrix:\n",dist)
    for i in range(22):
        for j in range(22):
            if i==j:
                dist[i][j] = 0
    
    """
    making condensed distance matrix of shape nC2
    """
    distArray = ssd.squareform(dist)
    #print(distArray)
    
    """
    making dendogram
    """
    Z = linkage(distArray)
    fig = plt.figure(figsize=(25, 10))
    
    
    dn = dendrogram(Z,labels=l)
    plt.show()
    
    """
    Elbow method - Hierarchical clustering
    """
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