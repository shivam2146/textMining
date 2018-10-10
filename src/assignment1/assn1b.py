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
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    print("tf idf",tfidf_model.shape)
    npm_tfidf = tfidf_model.todense()
    vocabulary = vectorizer.vocabulary_
    #print(vocabulary)
    sse = {}
    K = range(1,23)
    for k in K:
        km_model = KMeans(n_clusters=k)
        km_model.fit(tfidf_model)
        #print(km_model.labels_)
        sse[k] = km_model.inertia_
        
    #Elbow curve - K means
    
    plt.title("Elbow Method - Estimation of number of clusters")
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE error')
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()

    km_model = KMeans(n_clusters=5)
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
    
    
    df = DataFrame(cos_sm,columns=l,index=l)
    
    """
    Similarity matrix visualisation
    """
    plt.title('Similarity Matrix - Visualisation')
    ax = sns.heatmap(df,annot=True,annot_kws={"size":7},cmap="YlGnBu")
    ax.set(xlabel="Documents",ylabel="Documents")
    plt.show()
    
    """
    cosine dissimilarity matrix
    """
    dist = get_distance_matrix(npm_tfidf)
    #print("\n\ndistance matrix:\n",dist)
    for i in range(22):
        for j in range(22):
            if i==j:
                dist[i][j] = 0
    df1 = DataFrame(dist,columns=l,index=l)
    """
    distance matrix visualisation
    """
    plt.title('Distance Matrix - Visualisation')
    ax = sns.heatmap(df1,annot=True,annot_kws={"size":7},cmap="YlGnBu")
    ax.set(xlabel="Documents",ylabel="Documents")
    plt.show()
    
    """
    making condensed distance matrix of shape nC2
    """
    distArray = ssd.squareform(dist)
    print(distArray)
    
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
    max_d=.7
)
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