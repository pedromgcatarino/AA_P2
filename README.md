# AA_P2
 
Censoring:
Right censoring, how to deal with that?
- Use models that deal well with censored data (eg. RSF, AFT)

Unlabeled data:
How to deal with that? 
 - IsoMaps ? 
 - Semi-supervised learning
	-> Self Training
	-> Label Propagation

Missing values:
Missing values in train data and test data, how to deal with that?
- Option 1, do nothing and let the model handle missing values
- Option 2, discard rows/cols
- Option 3, imputation, but which?
	- Univariate imputation, multivarite imputation, NN imputation
