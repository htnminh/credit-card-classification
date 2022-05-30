import svm

filename = "clean_data.csv"
model3 = svm.SVM()
model3.getDataset(filename)
model3.dataPrepare(test_size= 0.2,dimention_reduct=True,k=28, feature_select=True, f_type = ('sfm',0.12), encoding = 21, oversampling=True)
model3.train(15,'auto')
model3.predict()
model3.confusion_matrix(True)
print(model3.accuraccy())
print(model3.f_score(0))