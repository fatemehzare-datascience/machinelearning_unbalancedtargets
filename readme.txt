The target coulmn of this dataset is unblanced in way that about 96% of the values are zero. 
To have a binary classification model with 0 and 1 classes, however, the accuracy is about 96 but the confusion matrix shows that there is
no true prediction for class 1.
To solve this issue, a resampling has been done on class 1 to increase the number of sample which has target 1 in coulmn 0.
The consfusion matrix after resampling shows a good classification for these dataset. Random forest has the best results for this dataset.
