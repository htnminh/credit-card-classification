import svm

filename = "clean_data.csv"
model4 = svm.SVM()
model4.getDataset(filename)
model4.dataPrepare(test_size= 0.2,dimention_reduct=True,k=28, feature_select=True, f_type = ('sfm',0.12), encoding = 21, oversampling=True)
model4.train(45,0.1)
model4.predict()
model4.confusion_matrix(True)
print(model4.accuraccy())
print(model4.f_score(0))