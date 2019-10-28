from sklearn.tree import DecisionTreeClassifier

class DTTrainer:

    def __init__(self,features_train,target_train):
        self.features=features_train
        self.target=target_train

    def train(self,rd_state):
        classifier = DecisionTreeClassifier(random_state=rd_state)
        self.classifier = classifier.fit(self.features,self.target)

    def test(self,ft_test):
        return self.classifier.predict(ft_test)

                                    
