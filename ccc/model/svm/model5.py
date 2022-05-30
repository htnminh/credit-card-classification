import svm

filename = "clean_data.csv"
model5 = svm.SVM()
model5.getDataset(filename)
model5.dataPrepare(test_size= 0.2,dimention_reduct=True,k=28, feature_select=True, f_type = ('sfm',0.12), encoding = 21, oversampling=False)
model5.train(10,0.1)
model5.predict()
model5.confusion_matrix(True)
print(model1.accuraccy())
print(model1.f_score(0))