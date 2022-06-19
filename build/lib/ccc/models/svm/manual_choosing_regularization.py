import svm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# When you don't know how to choose regularization so you decide to try each number,
# anyone know how to find Regularization in a correct paarrticular way please help me

def manual_finding_regularization(k =5,q=500,z = 5,t=10,type_take='gen', quantity=5):
    '''find regularization by experiment (try C from k to q, each C try t times and take mean
    then rank it with 6 criteria: accuracy, f-score for tp,tn, and difference of min and max( stability)
    return [C0,...,Cquantity]
    ----------------------
    k: start point try for C
    q: end pont try for C
    z: step between k,q
    t: times try for each C
    type_take: kind of result - acc: take C best accuraccy
                                f_score: take C best f_score
                                gen: teke C best (accuraccy, f_score, diffenrece)
    ----------------------'''
    info =[]
    s = svm.SVM()
    s.getDataset("clean_data.csv")

    #s.visualization(0, 'scatter', 1)
    for sample in range(k,q,z):
        trial_train = []
        for trial in range(t):
            s.dataPrepare(test_size= 0.2,dimention_reduct=True,k=28, feature_select=True, f_type = ('sfm',0.12), encoding = 21, oversampling=True)
            s.train(sample,0.1)
            s.predict()
            trial_train.append([s.accuraccy(), s.f_score(0), s.f_score(1)])
        trial_train = np.array(trial_train)
        info.append([sample]+list(trial_train.mean(axis=0)) + list(-trial_train.max(axis = 0) + trial_train.min(axis =0)))

    info =np.array(info)
    print(info)
    temp = pd.DataFrame(info[:,1:7]).rank(0,'dense',ascending=False).to_numpy()
    rank = np.concatenate((info[:,:1],temp, np.sum(temp,axis=1).reshape((k-q,1) )), axis =1)
    print(rank)
    if type_take == 'acc':
        result = np.concatenate((rank[:,1:2],rank[:,:1]),axis=1)
        result= result[np.argsort(result[:,0])]
        return result[:quantity,1]

    if type_take == 'f_score':
        result = np.concatenate((rank[:,2:3]+rank[:,3:4],rank[:,:1]),axis=1)
        result= result[np.argsort(result[:,0])]
        return result[:quantity,1]

    if type_take == 'gen':
        result = np.concatenate((rank[:,7:8],rank[:,:1]),axis=1)
        result= result[np.argsort(result[:,0])]
        print(np.concatenate((rank[:,7:8],rank[:,:1]),axis = 1))
        print(result)
        return result[:quantity,1]
    #s.plot_desicion_boundary()

if __name__ == "__main__":
    print(manual_finding_regulation(5,100,5,10,'gen',5))

