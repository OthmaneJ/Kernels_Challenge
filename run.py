import pandas as pd
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import pandas as pd
import time
from scipy import sparse



### Loading data #####
def load_sequence(kind='X', root='tr'):

    seqs =  [pd.read_csv('./data/%s%s%d.csv'%(kind, root, d)) for d in range(3)]

    if kind == 'X':
            df = pd.DataFrame(columns=['Id','seq'])
    else:
            df= pd.DataFrame(columns=['Id','Bound'])
    
    for seq in seqs:
        df = df.append(seq, ignore_index=True)
        
    return df


### Building the feature vectors 
def build_Kmers(sequence, size=5):

    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def get_features(sequences_train, sequences_test, size=5, rank= 4):

    df = sequences_train.copy()
    dg = sequences_test.copy()

    ## fitting the tf-idf vectorizer on the train dataset
    df['words'] = df.apply(lambda x: build_Kmers(x['seq'], size=size), axis=1)
    df = df.drop('seq', axis=1)

    texts = list(df['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(df.iloc[item,1])
    
        cv = TfidfVectorizer(ngram_range=(rank,rank))

    X = cv.fit_transform(texts)

    ## Using the tf-idf to transform the test dataset
    dg['words'] = dg.apply(lambda x: build_Kmers(x['seq'], size=size), axis=1)
    dg = dg.drop('seq', axis=1)

    texts = list(dg['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(df.iloc[item,1])

    Y = cv.transform(texts)

    return X,Y


## defining the classifier 
class Classifier():
    """
    distance based classifier for spectrum kernels
    """
    
    def __init__(self, clusters):
        self.clusters = clusters
    
    def compute_dist(self, X, Y):

        K_x = np.dot(X, X.T).toarray()
        K_y = np.dot(Y, Y.T).toarray()
        K_xy = np.dot(X, Y.T).toarray()
        return np.diag(K_x) - 2*K_xy.mean(axis=1) + K_y.mean()
    
    def predict(self, X):
        
        dists = np.array([self.compute_dist(X, cluster) for cluster in self.clusters])
        return dists.argmin(axis=0)

if __name__ == '__main__':

    # Loading data
    sequences_train = load_sequence()
    sequences_test = load_sequence(root='te')
    labels_train = load_sequence(kind='Y' ,root='tr')
    all_labels = labels_train.Bound.values.astype(int)

    sizes = [2, 3, 4, 5, 6, 7, 8] 
    predictions = []
    for size in tqdm(sizes):

        # building features
        X,Y = get_features(sequences_train, sequences_test, size=size)

        # building features
        clf= Classifier(clusters=[X[all_labels==0],X[all_labels==1]])
        test_pred = clf.predict(Y)
        predictions.append(test_pred)

    # ensemble learning using majority vote
    np.vstack(predictions)
    pred = (np.vstack(predictions).sum(axis=0)>(len(sizes)/2))*1

    ## saving the predictions on the test set
    Ids = np.array(range(3000))
    predictions = pd.DataFrame({'Id':Ids, 'Bound':pred.flatten()})
    predictions.to_csv('./predictions/Yte.csv', index=False)

    print('predictions saved')






