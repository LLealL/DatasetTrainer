from sklearn.neural_network import MLPClassifier

class MLPTrainer:

    def __init__(self,ft_train,tg_train):
        self.features=ft_train
        self.target=tg_train

    def train(self,layers,rd_state,epochs,learn_rate):
        classifier = MLPClassifier(hidden_layer_sizes=layers,random_state=rd_state,max_iter=epochs,learning_rate_init=learn_rate)
        self.classifier = classifier.fit(self.features,self.target)

    def test(self,ft_test):
        return self.classifier.predict(ft_test)
