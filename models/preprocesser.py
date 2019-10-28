from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

class PreProcesser:

    def __init__(self,features,target):
        self.features=features
        self.target=target

    def getNormalized(self,state,size):
        ft_train,ft_test,tg_train,tg_test = train_test_split(self.features,self.target,train_size=size,stratify=self.target,random_state=state)
        norm = Normalizer()
        ft_train_n = norm.fit_transform(ft_train)
        ft_test_n= norm.fit_transform(ft_test)
        return ft_train_n, ft_test_n;

    def getStdScaled(self,state,size):
        ft_train,ft_test,tg_train,tg_test=train_test_split(self.features,self.target,train_size=size,stratify=self.target,random_state=state)
        scaler = StandardScaler()
        ft_train_std = scaler.fit_transform(ft_train)
        ft_test_std = scaler.fit_transform(ft_test)

        return ft_train_std,ft_test_std;

    def getTargets(self,state,size):
        ft_train,ft_test,tg_train,tg_test = train_test_split(self.features,self.target,train_size=size,stratify=self.target,random_state=state)
        return tg_train,tg_test;

        
