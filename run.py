import pandas as pd
import time
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import Levenshtein

from cvxopt import matrix, spmatrix, solvers
from scipy.special import expit
import numpy as np
import sys
import pandas as pd
import time
from scipy import sparse



### Lodding data #####
def load_sequence(kind='X', root='tr', number=3):
    """
    Load DNA sequences
    """
    seqs =  [pd.read_csv('./data/%s%s%d.csv'%(kind, root, d)) for d in range(number)]
    
    if kind == 'X':
            df = pd.DataFrame(columns=['Id','seq'])
    else:
            df= pd.DataFrame(columns=['Id','Bound'])
    
    for seq in seqs:
        
        df = df.append(seq, ignore_index=True)
        
    return df


### Getting the K-mers ###
def getKmers(sequence, size=5):
    """
    Builds kmers
    """
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def get_features(sequences_train, sequences_test, size=5, normed=False, rang=(4,4)):

    df = sequences_train.copy()
    dg = sequences_test.copy()

    df['words'] = df.apply(lambda x: getKmers(x['seq'], size=size), axis=1)
    df = df.drop('seq', axis=1)

    texts = list(df['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(df.iloc[item,1])
    
    if normed:
        cv = TfidfVectorizer(ngram_range=rang)
    else:
        cv = CountVectorizer(ngram_range= rang)
    X = cv.fit_transform(texts)


    dg['words'] = dg.apply(lambda x: getKmers(x['seq'], size=size), axis=1)
    dg = dg.drop('seq', axis=1)

    texts = list(dg['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(df.iloc[item,1])

    Y = cv.transform(texts)

    return X,Y


## defining the classifier 
class KernelNC():
    """
    distance based classifier for spectrum kernels
    """
    
    def __init__(self, classes):
        self.classes = classes
    
    def compute_dist(self, X, Y):

        K_x = np.dot(X, X.T).toarray()
        K_y = np.dot(Y, Y.T).toarray()
        K_xy = np.dot(X, Y.T).toarray()


        
        return np.diag(K_x) - 2*K_xy.mean(axis=1) + K_y.mean()
    
    def predict(self, X):
        
        dists = np.array([self.compute_dist(X, classe) for classe in self.classes])
        return dists.argmin(axis=0)
    
    def score(self, X, y):
        y__ = self.predict(X)
        return 100*(y__==y).mean()


if __name__ == '__main__':

    sequences_train = load_sequence(number=3)
    sequences_test = load_sequence(number=3, root='te')
    labels_train = load_sequence(kind='Y' ,root='tr', number=3)
    all_labels = labels_train.Bound.values.astype(int)

    sizes =  [2,3, 4, 5, 6]
    predictions = []
    for size in tqdm(sizes):
        # X = build_spectrum_kernels(sequences_train,size=size)
        # Y = build_spectrum_kernels(sequences_test,size=size)
        # X = sparse.csr_matrix(X)
        # Y = sparse.csr_matrix(Y)

        X,Y = get_features(sequences_train, sequences_test, size=size, normed=False, rang=(4,4))

        clf= KernelNC(classes=[X[all_labels==0],X[all_labels==1]])
        test_pred = clf.predict(Y)
        predictions.append(test_pred)

        # train_pred = clf.predict(X)
        # predictions.append(train_pred)
        

    np.vstack(predictions)
    pred = (np.vstack(predictions).sum(axis=0)>(len(sizes)/2))*1

    ## saving the predictions on the test set
    Ids = np.array(range(3000))
    predictions = pd.DataFrame({'Id':Ids, 'Bound':pred.flatten()})
    # predictions.to_csv('./predictions/Yte.csv', index=False)

    print('predictions saved')






