The target column of this dataset is unbalanced in a way that about 96% of the values are zero.
To have a binary classification model with 0 and 1 classes, however, the accuracy is about 96, but the confusion matrix shows that there is no accurate prediction for class 1. 
To address this issue, a resampling has been done in class 1 to increase the number of the sample, which has target 1 in column 0.
The confusion matrix after resampling shows a good classification for this dataset. The random forest has the best results for this dataset.
