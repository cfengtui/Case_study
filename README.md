## Introduction

This data set consists consumer mortages and delinquencies. The objective of this case study is to estimate the chance that a consumer will be delinquent on his mortgage on his mortgage repayments in the next month. 

## Data

The data is displayed on loan part level.

## How to excute the jupyter notebook
### Required packages
glob  
numpy  
pandas  
matplotlib  
seaborn  
xgboost  

### Data 
Copy the following four csv files to the working directory. The script uses a glob function to find all files with format data*.csv, please make sure that there are no other .csv files with similar names in this directory.
data01.csv  
data02.csv  
data03.csv  
data04.csv  

The data_cleaned file consists cleaned data at consumer level, this data has been cached to save some computing time. It will take 10-20 minutes for the notebook to generate the cleaned data set, and would not be re-generated after the first run. 

## Modelling Procedure

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
- According to the correlation plot, we see that there are highly correlated variables, to prevent from multicollinearity (that may cause problems for logistic regression classifier), we select 12 numerical features and 2 categorical features (have been convered to numerical features by one-hot encoding).
- We take the one-step forward forecasting approach: set one if the arrears balance of the next month is positive, otherwise set zero. 
- We split 80% data as training/cross-validation set, and 20% as testing set. We ignore the time dimension, and treat each record as i.i.d. 
  Since we have imbalanced dataset (much more zeros than ones), We use the option "stratify" to let the positive/negative propotion remain approximately the same in each set.
  
### Classifier selection
Since as a debt management company, we forcus mainly on the positive records, and want to keep the balance of recall (TP/(TP+FN)) and precision (TP/(TP+FP)). We give more weight on recall to minimize the false-negative rate as much as possible, therefore we choose f2 score as our scroing matrix. 

We use sklearn cross-validation pipeline and f2 score to evaluate the performance of three classifiers: logistic regression, random forest, and xgboost. It turns out that xgboost produces the best results.

### Grid search parameters
The xgboost produce relatively stable results, we only tune two hyperparameters to further improve its performance:
- scale_pos_weight: the recommand value is sum(y_train==0)/sum(y_train==1), which is around 38 in our data, we search in the range of [5,10,40]  
- max_depth: this parameter controls the sophistication of the model, we choose the range [3,4,5]. A too high value may cause overfitting to our training set.  

### Fit the best model
We plot the precision-recall curve to demonstrate the performance of the final model, and move the threshold to maximize the f2 score. It turns out that the optimal threshold of 0.56 is not far from the default value of 0.5. 

We compare the performance (recall, precision and f2) of the xgboost model with two benchmark models: persistence model (always use this month deliquency to forecast next month), and naive model (if a consumer has ever positive value in his/her arrears balance, then we keep forecasting positive values in the future). Xgboost model produces much better results than the two benchmark models:

| Result        | Recall        | Precision  | F2 score  |
| ------------- |:-------------:| ----------:| ---------:|
| Xgboost       | 0.67          | 0.61       | 0.66      |
| Persistence   | 0.54          | 0.57       | 0.55      |
| Naive         | 1.00          | 0.09       | 0.34      |

### Make the final prediction
We calculate the probability of delinquency of each consumer in the next month. 

### Bonus Question
For all consumers that have ever being delinquent, the probability will be one.
For other consumers, for each individual i,  calculate the months to matuarity m_i, the model has produced the delinquent probability p_i, then the probability of delinquent at any time in the mortgage lifetime is: 1-(1-p_i)**m_i

The above calcultion is based on the assumption that the p_i remains the same in the future. We can of course simulate all the future data of X to calculate p_i_t, and adjust above formula accordingly. 
