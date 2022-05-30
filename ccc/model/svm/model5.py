import svm

filename = "clean_data.csv"
model1 = svm.SVM()
model1.getDataset(filename)
model1.dataPrepare(test_size= 0.2,dimention_reduct=True,k=28, feature_select=True, f_type = ('sfm',0.12), encoding = 21, oversampling=False)
model1.train(10,0.1)
model1.predict()
model1.confusion_matrix(True)
print(model1.accuraccy())
print(model1.f_score(0))