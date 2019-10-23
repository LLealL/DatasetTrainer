import pandas as pd
import warnings
import threading
import logging
import logging.handlers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer
from os import listdir
from sklearn.tree.export import export_text

def warn(*arg,**kwargs):
    pass


def createPerceptronThread(features,target,dataset,log):
    maior_acc =0
    melhor_taxa=0
    melhor_iter=0
    melhor_random_state=0
    norm_or_std = 0
    my_recall=0
    my_precision=0
    my_matthews=0
    for i in range(50):    
        feature_train, feature_test, target_train, target_test = train_test_split(features,target,train_size=0.7, stratify=target,random_state=0)
        scaler = StandardScaler()
        feature_train_std = scaler.fit_transform(feature_train)
        feature_test_std = scaler.fit_transform(feature_test)

        normalizer = Normalizer()
        feature_train_norm =  normalizer.fit_transform(feature_train)
        feature_test_norm = normalizer.fit_transform(feature_test)
        
        j=25
        while j<500:
            k=0.05
            while k<0.5:
                pcptron_std = Perceptron(max_iter=j, random_state= i, tol=k)
                pcptron_std = pcptron_std.fit(feature_train_std,target_train)
                predict_std = pcptron_std.predict(feature_test_std)

                acc = accuracy_score(predict_std, target_test)
                if acc>maior_acc:
                    maior_acc=acc
                    melhor_taxa=k
                    melhor_iter=j
                    melhor_random_state=i
                    norm_or_std=1
                    my_recall = recall_score(target_test, predict_std)
                    my_precision= precision_score(target_test,predict_std)
                    my_matthews= matthews_corrcoef(target_test, predict_std)  
                    tn,fp,fn,tp =confusion_matrix(target_test, predict_std).ravel()

                pcptron_norm = Perceptron(max_iter=j, random_state=i, tol=k)
                pcptron_norm = pcptron_norm.fit(feature_train_norm,target_train)
                predict_norm = pcptron_norm.predict(feature_test_norm)
                
                acc = accuracy_score(predict_norm,target_test)
                if acc>maior_acc:
                    maior_acc=acc
                    melhor_taxa=k
                    melhor_iter=j
                    melhor_random_state=i
                    norm_or_std=0
                    my_recall = recall_score(target_test, predict_norm)
                    my_precision= precision_score(target_test,predict_norm)
                    my_matthews= matthews_corrcoef(target_test, predict_norm) 
                    tn,fp,fn,tp = confusion_matrix(target_test, predict_norm).ravel()
                    
                k+=0.05
                
            j= j + 25
        
    message = "\n -----------------------"    
    message+= "\n Perceptron: "+dataset
    message+= "\n Melhor Resultado:"
    message+= "\n Acurracy= " + str(maior_acc)
    message+= "\n Precision= " +str(my_precision)
    message+= "\n Recall= " +str(my_recall)
    message+= "\n Matthews Coeficient= " + str(my_matthews)
    message+= "\n True Negative: " + str(tn) + " True Positive:" + str(tp)
    message+= "\n False Negative: "+ str(fn) + " False Positive: " + str(fp)
    message+= "\n random_state= " + str(melhor_random_state)
    message+= "\n iterações= " + str(melhor_iter)
    message+= "\n taxa de aprendizado= " + str(melhor_taxa)
    if norm_or_std==1:
        message += "\n Com StandardScaler"
    else:
        message += "\n Com Normalizer"

    addToLogger(log,message)



        
        

def createDTthread(features,target,dataset,log):
    maior_acc=0;
    melhor_state=0;
    norm_or_std=0;
    my_recall=0
    my_precision=0
    my_matthews=0
    for i in range(50):
        ft_train , ft_test, tg_train, tg_test = train_test_split(features,target,train_size=0.7, stratify=target, random_state=0)

        scaler=  StandardScaler()
        ft_train_std= scaler.fit_transform(ft_train)
        ft_test_std= scaler.transform(ft_test)

        n = Normalizer()
        ft_train_norm= n.fit_transform(ft_train)
        ft_test_norm = n.transform(ft_test)

        decision_tree_std = DecisionTreeClassifier(random_state=i)
        decision_tree_std = decision_tree_std.fit(ft_train_std, tg_train)
        prediction_std = decision_tree_std.predict(ft_test_std)

        acc= accuracy_score(prediction_std,tg_test)
        if acc>maior_acc:
            maior_acc=acc
            maior_state=i
            norm_or_std=1
            my_recall = recall_score(tg_test, prediction_std)
            my_precision= precision_score(tg_test,prediction_std)
            my_matthews= matthews_corrcoef(tg_test, prediction_std)  
            tn,fp,fn,tp =confusion_matrix(tg_test, prediction_std).ravel()
            
        
        decision_tree_norm = DecisionTreeClassifier(random_state=i)
        decision_tree_norm = decision_tree_norm.fit(ft_train_norm,tg_train)
        prediction_norm = decision_tree_norm.predict(ft_test_norm)
            
        acc= accuracy_score(prediction_norm,tg_test)
        if acc>maior_acc:
            maior_acc=acc
            maior_state=i
            norm_or_std=0
            my_recall = recall_score(tg_test, prediction_norm)
            my_precision= precision_score(tg_test,prediction_norm)
            my_matthews= matthews_corrcoef(tg_test, prediction_norm)  
            tn,fp,fn,tp =confusion_matrix(tg_test, prediction_norm).ravel()

    message = "\n -----------------------"
    message+= "\n Decision Tree: "+dataset
    message+= "\n Melhor Resultado:"
    message+= "\n Acurracy= " + str(maior_acc)
    message+= "\n Precision= " +str(my_precision)
    message+= "\n Recall= " + str(my_recall)
    message+= "\n Matthews Coeficient= " + str(my_matthews)
    message+= "\n True Negative: " + str(tn) + " True Positive:" + str(tp)
    message+= "\n False Negative: "+ str(fn) + " False Positive: " + str(fp)
    if norm_or_std==1:
        message += "\n Com StandardScaler"
    else:
        message += "\n Com Normalizer"

    addToLogger(log,message)
            


def addignore(dataset):
    f = open("trainIgnore.txt", "a")
    s = dataset +"\n"
    f.write(s)
    f.close()

def readIgnores():
    f = open("trainIgnore.txt", "r")
    ignores = []
    while True :
        x=f.readline()
        x= x.rstrip('\n')
        if x is "":
            break
        ignores.append(x)
    return ignores

def createLogger():
    my_logger = logging.getLogger('MyLogger')
    my_logger.setLevel(logging.DEBUG)

    handler = logging.handlers.RotatingFileHandler('results.out')

    my_logger.addHandler(handler)
    return my_logger

def addToLogger(Logger,message):
    Logger.debug(message)
    









warnings.warn=warn

dir = 'datasets/'
output = 'result.csv'

names= []
logger = createLogger()
ignores = readIgnores()
for file in listdir(dir):
    print("preparing...",file)
    if file in ignores:
        print("skipping ",file)
        continue
    names.clear()
    with open(dir+file, 'rt') as in_file:
        for line in in_file:
            if line.startswith("@inputs"):
                for word in line.split(" "):
                    if word != '@inputs':
                        names.append(word.replace('\n', ''))
                names.append("classes")
            if line.startswith("@data"):
                break

    data = pd.read_csv(dir+file, comment='@' , header=None)
    encoder = LabelEncoder()
    data = data.apply(encoder.fit_transform)
    ultimaColuna = len(names)-1
    
    features = data.iloc[:, 0:ultimaColuna]
    tags = data.iloc[:, -1]

    ptron= threading.Thread(target=createPerceptronThread, args=(features,tags,file,logger))
    dtree= threading.Thread(target=createDTthread,args=(features,tags,file,logger))
    ptron.start()
    dtree.start()
    ptron.join() 
    dtree.join()
    addignore(file)
    addToLogger(logger,"################################")




