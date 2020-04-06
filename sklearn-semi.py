import numpy as np
from utils import maskData
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation

def runLP():
        
    #IMPORT DATASETS
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target
    
    #UNLABEL 95% OF THE DATASET
    masked_labels = maskData(labels, 0.05)
    
    #RUN THE MODEL
    model = LabelPropagation()
    model.fit(data, masked_labels)
    pred = np.array(model.predict(data))
    
    #SEPARATE PREDICTED SAMPLES
    labels = np.array(labels[masked_labels == -1]).astype(int)
    pred = pred[masked_labels == -1]
    
    #PRINT CONFUSION MATRIX
    print(confusion_matrix(labels, pred))
    
if __name__ == '__main__':
    runLP()
