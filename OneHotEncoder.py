import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.tags=[]

    def fit(self,X):
        """Converts list of labels into unique list and stores in self.tags.

        :param X: list of labels
        :return: None
        """


        self.tags=[]
        for x in X:
            if x not in self.tags:
                self.tags.append(x)
                
        pass

    def fit_transform(self,X):
        """Calls fit and transform methods respectively with X.

        :param X: list of labels
        :return: numpy array of one-hot vectors for each element in X
        """


        OneHotEncoder.fit(self, X)
        result=OneHotEncoder.transform(self,X)
        return result
    
        pass

    def transform(self, X):
        """Converts each element in the list into their one-hot representations

        :param X: list of labels
        :return: numpy array of one-hot vectors for each element in X
        """


        oneHotVector = np.zeros((len(X),len(self.tags)),dtype=int)
        for i in range(0,len(X)):
            if(X[i] in self.tags):
                myIndex = self.tags.index(X[i])
                oneHotVector[i][myIndex]=1
        return oneHotVector
    
        pass

    def get_feature_names(self):
        """Returns the tags
        :return: tags
        """


        return self.tags
    
        pass

    def decode(self, one_hot_vector):
        """Decodes given one-hot-vector into its value.

        :param one_hot_vector: numpy array for one-hot-vector
        :return: corresponding element in self.tags
        """


        for i in range(0, len(one_hot_vector)):
            if one_hot_vector[i]==1:
                return self.tags[i]
            
        pass

if __name__=="__main__":
    o = OneHotEncoder()
    train_labels = ["Action","Comedy","Crime","Comedy","Crime","Musical","Action"]
    test_labels = ["Comedy","Action","Crime","Musical","Crime","War"]
    train_one_hot_vectors = o.fit_transform(train_labels)
    test_one_hot_vectors = o.transform(test_labels) 
    print o.get_feature_names()
    print train_one_hot_vectors
    print test_one_hot_vectors
    one_hot_vector = np.array([0,1,0,0])
    print o.decode(one_hot_vector)
