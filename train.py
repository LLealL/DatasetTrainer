import pandas as pd
import warnings
import threading
import logging
import logging.handlers
import tqdm
from models.mlptrainer import MLPTrainer
from models.dttrainer import DTTrainer
from models.preprocesser import PreProcesser
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.impute import SimpleImputer
from os import listdir
from sklearn.tree.export import export_text

def warn(*arg,**kwargs):
    pass


def createMLPThread(features,target,dataset,log):
    preproc = PreProcesser(features,target)
    ft_train_norm, ft_test_norm = preproc.getNormalized(state=0, size=0.7)
    tg_train,tg_test = preproc.getTargets(state=0, size=0.7)
    layers = [(10,),(20,),(50,),(10,10),(20,10),(50,10)]
    epocas =[25,50,100]
    rates = [0.01, 0.05]
    message = "\nMLP :"+ dataset
    addToLogger(log,message)
    bar= tqdm.tqdm(total=10*len(layers)*len(epocas)*len(rates),desc=dataset+" MLP")
    for i in range(10):
        trainer= MLPTrainer(ft_train_norm,tg_train)
        message ="\nrandom_state: "+str(i)
       # layersBar = tqdm.tqdm(total=len(layers))
        for layer in layers:
            message+="\nlayers: "+str(layer)
           # epocasBar = tqdm.tqdm(total = len(epocas))
            for epoca in epocas:
                message+="\nepochs: " + str(epoca)
                for rate in rates:
                    message+="\nlearning_rate: "+str(rate)
                    trainer.train(layers=layer,rd_state=i,epochs=epoca,learn_rate=rate)
                    predict = trainer.test(ft_test_norm)
                    message+= "\nAccuracy: " +str(accuracy_score(tg_test,predict))
                    bar.update(1)
           #     epocasBar.update(1)
          #  epocasBar.close()
         #   layersBar.update(1)
        #layersBar.close()
        #bar.update(1)
        addToLogger(log,message)
    bar.close()
    
    

def createPerceptronThread(features,target,dataset,log):
    maior_acc =0
    melhor_taxa=0
    melhor_iter=0
    melhor_random_state=0
    norm_or_std = 0
    my_recall=0
    my_precision=0
    my_matthews=0
    bar = tqdm.tqdm(total=30)
    for i in range(30):
        preproc = PreProcesser(features,target)
        ft_train_std , ft_test_std = preproc.getStdScaled(state=0,size=0.7)
        ft_train_norm, ft_test_norm = preproc.getNormalized(state=0,size=0.7)
        tg_train,tg_test = preproc.getTargets(state=0,size=0.7)

        bar.update(1)
        j=0
        while j<100:
            k=0.05
            j= j + 1
            while k<0.5:
                trainer_std = PerceptronTrainer(ft_train_std,tg_train)
                trainer_norm = PerceptronTrainer(ft_train_norm,tg_train)
                trainer_std.train(its=j , rd_state=i)
                trainer_norm.train(its=j , rd_State=i)
                predict_std = trainer_std.test(ft_test_std)
                predict_norm = trainer_std.test(ft_test_norm)
                
                acc = accuracy_score(target_test, predict_std)
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
                
                acc = accuracy_score(target_test,predict_norm)
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
        preproc = PreProcesser(features,target)
        ft_train_std , ft_test_std = preproc.getStdScaled(state=0,size=0.7)

        ft_train_norm, ft_test_norm = preproc.getNormalized(state=0,size=0.7)

        tg_train,tg_test = preproc.getTargets(state=0,size=0.7)
        
        trainer_std = DTTrainer(ft_train_std,tg_train)
        trainer_norm = DTTrainer(ft_train_norm,tg_train)
        trainer_std.train(rd_state=i)
        trainer_norm.train(rd_state=i)
        prediction_std = trainer_std.test(ft_test_std)
        prediction_norm = trainer_norm.test(ft_test_norm)
        
        acc= accuracy_score(tg_test,prediction_std)
        if acc>maior_acc:
            maior_acc=acc
            maior_state=i
            norm_or_std=1
            my_recall = recall_score(tg_test, prediction_std)
            my_precision= precision_score(tg_test,prediction_std)
            my_matthews= matthews_corrcoef(tg_test, prediction_std)  
            tn,fp,fn,tp =confusion_matrix(tg_test, prediction_std).ravel()
                        
        acc= accuracy_score(tg_test_norm,prediction_norm)
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
    lines = f.readlines()

    for line in lines:
        x = line.rstrip('\n')
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

names= []
logger = createLogger()
ignores = readIgnores()
for file in listdir(dir):
    print("\npreparing...",file)
    if file in ignores:
        print("\nskipping ",file)
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
   # dtree= threading.Thread(target=createDTthread,args=(features,tags,file,logger)  
   # mlpThread= threading.Thread(target=createMLPThread,args=(features,tags,file,logger))

    createMLPThread(features,tags,file,logger)
    addignore(file)
    addToLogger(logger,"################################")




