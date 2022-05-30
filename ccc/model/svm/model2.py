import svm

filename = "clean_data.csv"
model2 = svm.SVM()
model2.getDataset(filename)
model2.dataPrepare(test_size= 0.2,dimention_reduct=True,k=28, feature_select=True, f_type = ('sfm',0.12), encoding = 21, oversampling=True)
model2.train(80,'auto')
model2.predict()
model2.confusion_matrix(True)
print(model1.accuraccy())
print(model1.f_score(0))
