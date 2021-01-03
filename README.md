## Introduction

This data set consists consumer mortages and delinquencies. The objective of this case study is to estimate the chance that a consumer will be delinquent on his mortgage on his mortgage repayments in the next month. 

## Data

The data is about 3000 consumers with 415229 records, displayed on loan part level.
-	DebtID: Unique identity of a loan part   
-	ConsumerID: Unique identity of a consumer  
-	LoanAgeR: Loan age in months  
-	InterestRateB: fixed interest rate in percentage  
-	NumberMonthsInArrears: Number of months a consumer payment is in arrears  
-	CurrentBalance: Current principal amount  
-	EstDisposableIncome: Estimates disposable income of the consumer  
-	ArrearsBalance: The amount in arrears as of the reporting date  
-	TotalExposure: Total mortgage amount including all mortgage parts  
-	IndexedTotalIncome: consumer income indexed to latest income index  
-	Original Balance: Mortgage principal amount at the time of origination  
-	LoanOriginationDate: Mortgage origination date  
-	MaturityDateR: Mortgage maturity date  
-	ReportDateB: Reporting date  
-	ConsumerAge: Age of consumer in years  
-	PropertyRegion: Region in which the property is located  
-	OriginalPropertyValue: Property value at mortgage origination assigned to a loan part. The total value of a property is the sum of property values of all loan parts   
-	ForeclosureValue: Foreclosure value of property after factoring in costs   
-	PropertyIndexFactor: Property value index factor applicable to both property value and foreclosure value.   

## Python Notebook Introduction 
### Required packages
glob  
numpy  
pandas  
matplotlib  
seaborn  
xgboost  

### Loading and preprocessing the data
Load and merge four .csv files into pandas dataframe format.
convert date formats

### Imputing missing values
There are 7 columns out of 19 contain missing values, we choose not to drop any records, but instead impute values. 

### Clean data
In some columns there are errors in the data: for example, ConsumerID 2166453 on 2016-11-30, the foreclosure value of a loan is 6.378, however the normal value should be 62942.6. We corrected those kinds of errors before we visualize the data and put the feature into our model.

### Prepare data
- Since our prediction will be on consumer level, we need to transform our loan part level data to consumer level first.
- We add some extra features: first difference of arears balnce and its lags, payment to income ratios and their lags, report month (one-hot), ect
- According to the correlation plot, we see that there are highly correlated variables, to prevent from multicollinearity (that may cause problems for logistic regression classifier), we select 16 features.
- We take the one-step forward forecasting approach: set one if the arrears balance of the next month is positive, otherwise set zero. 
- We split 80% data as training/cross-validation set, and 20% as testing set. We ignore the time dimension, and treat each record as i.i.d. 
  Since we have imbalanced dataset (much more zeros than ones), We use the option "stratify" to let the positive/negative propotion remain approximately the same in each set.
  
### Classifier selection
Since as a debt management company, we forcus mainly on the positive records, and want to keep the balance of recall (TP/(TP+FN)) and precision (TP/(TP+FP)). We give more weight on recall to minimize the false-negative rate as much as possible, therefore we choose f2 score as our scroing matrix. 

We use sklearn cross-validation pipeline and f2 score to evaluate the performance of three classifiers: logistic regression, random forest, and xgboost. It turns out that xgboost produces the best results.

### Grid search parameters
The xgboost produce relatively stable results, we only tune three hyperparameters to further improve its performance:
- scale_pos_weight: the recommand value is sum(y_train==0)/sum(y_train==1), which is around 38 in our data, we search in the range of [5,10,15,20,40]  
- max_depth: this parameter controls the sophistication of the model, we choose the range [2,3,4,5]. A too high value may cause overfitting to our training set.  
- n_estimators: this parameter is related to max_depth, the higher the max_depth, the lower the n_estimator may require.  


### Fit the best model
We plot the precision-recall curve to demonstrate the performance of the final model, and move the threshold to maximize the f2 score. It turns out that the optimal threshold of 0.56 is not far from the default value of 0.5. 

We compare the performance (recall, precision and f2) of the xgboost model with two benchmark models: persistence model (always use this month deliquency to forecast next month), and naive model (if a consumer has ever positive value in his/her arrears balance, then we keep forecasting positive values in the future). Xgboost model produces much better results than the two benchmark models:

| Result        | Recall        | Precision  | F2    |
| ------------- |:-------------:| ----------:| -----:|
| Xgboost       | 0.72          | 0.56       |0.68   |
| Persistence   | 0.57          | 0.60       |0.58   |
| Naive         | 1.00          | 0.10       |0.35   |

### Make the final prediction
We calculate the probability of delinquency of each consumer in the next month. 
