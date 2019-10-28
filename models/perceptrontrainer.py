from sklearn.linear_model import Perceptron

class PerceptronTrainer():

    def __init__(self,ft_train,tg_train):
        self.features=ft_train
        self.target=tg_train

    def train(its, rd_state):
        classifier = Perceptron(max_iter=its,random_state=rd_state)
        self.classifier=classifier.fit(self.features,self.target)


    def test(ft_test):
        return self.classfier.predict(ft_test)
