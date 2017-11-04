# Practical Machine Learning - Final Project
Wiktoria Urantowka  
11/3/2017  



##1. Introduction##

The goal of this analysis is to build a classifier algorith that predicts if a performed excercise was done correctly or not and if not in what way.Classification is build using the data from accelerometers placed on the belt, forearm, arm, and dumbell as well as responce variables: classes A, B, C, D and E, class A corresponding to a correct execution of the exercise, and the remaining ones to common mistakes. 

##2. Loading and partitioning data##

```r
trainURL <-"http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
data_training <-read.csv(url(trainURL), na.strings=c("NA","#DIV/0",""))
data_validation <-read.csv(url(testURL), na.strings=c("NA","#DIV/0",""))
dim(data_training)
```

```
## [1] 19622   160
```

```r
set.seed(1001)
library(caret)
library(randomForest)
inTrain<-createDataPartition(data_training$classe, p=0.8, list = FALSE)
training <-data_training[inTrain,]
testing <-data_training[-inTrain,]
```


```r
dim(data_training) 
```

```
## [1] 19622   160
```

```r
dim(training) 
```

```
## [1] 15699   160
```

```r
dim(testing)
```

```
## [1] 3923  160
```

##3. Cleaning Data##

3.1 Remove variables with very low variance


```r
lowvar <- nearZeroVar(training, saveMetrics = TRUE)
training <-training[-lowvar$nzv==FALSE]
dim(training)
```

```
## [1] 15699   120
```

3.2 Remove variables with high percentage of missing values

a. Compute percentage of NA for each variable

```r
na_count <- sapply(training, function(y) sum(length(which(is.na(y)))))
na_count <-data.frame(colnames(training),na_count)
na_count[,2]<-na_count$na_count/dim(training)[1]
na_percent<-na_count
head(na_percent, n=12)
```

```
##                        colnames.training.  na_count
## X                                       X 0.0000000
## user_name                       user_name 0.0000000
## raw_timestamp_part_1 raw_timestamp_part_1 0.0000000
## raw_timestamp_part_2 raw_timestamp_part_2 0.0000000
## cvtd_timestamp             cvtd_timestamp 0.0000000
## num_window                     num_window 0.0000000
## roll_belt                       roll_belt 0.0000000
## pitch_belt                     pitch_belt 0.0000000
## yaw_belt                         yaw_belt 0.0000000
## total_accel_belt         total_accel_belt 0.0000000
## kurtosis_roll_belt     kurtosis_roll_belt 0.9792343
## kurtosis_picth_belt   kurtosis_picth_belt 0.9792343
```

```r
sum(na_percent$na_count==0)/length(na_percent[,2])
```

```
## [1] 0.4916667
```
Half of variables have more then 95 % of NA. They are useless and will be removed from the sample

b. Removing variables with prevalence of NA

```r
columns_left <-subset(na_percent, na_count == 0)[,1]
cleantraining <-training[,as.character(columns_left)]
dim(cleantraining)
```

```
## [1] 15699    59
```

3.3 Reducing Dimentions

a. Removing variables that have clearly nothing to do with the the response variable

```r
cleantraining <-cleantraining[c(-1)][c(-1)][c(-1)][c(-1)][c(-1)]
```

b. Dealing with collinearity 

```r
##correlating numerical variables
set_for_cor <-cleantraining[c(-54)]
corr <-cor(set_for_cor)
hc<-findCorrelation(corr,cutoff = 0.6, verbose=FALSE, names=FALSE)
reduced_data <-cleantraining[,-c(hc)]
dim(reduced_data)
```

```
## [1] 15699    29
```
findCorrelation() function looks up variables whose correlation corresponds to the defined cutoff (here 0.6) and out of the paire of correlated variables selects the one with the higher correlation with the remaining ones, so facilitates the selection of those to remove. Another 30 variables are dropped.
Out of 160 candidate predictors, after some cleanup we remain with 29 and are ready to proceed to the prediction.

##4 Prediction using Decision Tree##  
4.1 Model Fitting on the training subsample

```r
set.seed(1001)
library(rpart)
modFit1 <-rpart(classe~ .,data=reduced_data, method="class")
plot( modFit1,uniform=TRUE, main="Decision Tree")
text(modFit1, cex=0.5)
```

![](FinalProject_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

4.2 Prediction on the testing subsample and assessing the out of sample error
 

```r
predictions1 <-predict( modFit1, testing, type="class")
confusionMatrix<- confusionMatrix(predictions1, testing$classe)
confusionMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 916  26   4  10  13
##          B 123 573  92 217  61
##          C  46  88 559  49  23
##          D   1  38  21 328  59
##          E  30  34   8  39 565
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7497          
##                  95% CI : (0.7358, 0.7632)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6843          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8208   0.7549   0.8173  0.51011   0.7836
## Specificity            0.9811   0.8442   0.9364  0.96372   0.9653
## Pos Pred Value         0.9453   0.5375   0.7307  0.73378   0.8358
## Neg Pred Value         0.9323   0.9349   0.9604  0.90938   0.9520
## Prevalence             0.2845   0.1935   0.1744  0.16391   0.1838
## Detection Rate         0.2335   0.1461   0.1425  0.08361   0.1440
## Detection Prevalence   0.2470   0.2717   0.1950  0.11394   0.1723
## Balanced Accuracy      0.9010   0.7996   0.8768  0.73691   0.8745
```

```r
plot(confusionMatrix$table, main="Decision Tree Confusion Matrix (Accuracy=0.74)")  
```

![](FinalProject_files/figure-html/unnamed-chunk-9-1.png)<!-- -->
    
Accuracy of 0.74 indicated that the model probably make sence but is not very high either. Let's try random forest to see if we can get any better. As it uses boosing to build trees and select variables, it is expected thet the accuracy will increase.

##5 Prediction using Random Forest##
5.1 Model Fitting on the training subsample

```r
set.seed(1001)
modFit2 <-randomForest(classe~ .,data=reduced_data)
```
5.2 Prediction on the testing subsample and assessing the out of sample error

```r
predictions2 <-predict( modFit2, testing, type="class")
confusionMatrix2<- confusionMatrix(predictions2, testing$classe)
confusionMatrix2
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    0    0    0    0
##          B    0  759    5    1    0
##          C    0    0  679    5    0
##          D    0    0    0  634    0
##          E    1    0    0    3  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9962          
##                  95% CI : (0.9937, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9952          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   1.0000   0.9927   0.9860   1.0000
## Specificity            1.0000   0.9981   0.9985   1.0000   0.9988
## Pos Pred Value         1.0000   0.9922   0.9927   1.0000   0.9945
## Neg Pred Value         0.9996   1.0000   0.9985   0.9973   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1935   0.1731   0.1616   0.1838
## Detection Prevalence   0.2842   0.1950   0.1744   0.1616   0.1848
## Balanced Accuracy      0.9996   0.9991   0.9956   0.9930   0.9994
```

```r
plot(confusionMatrix2$table, main="Random Forest Confusion Matrix (Accuracy=0.99)")  
```

![](FinalProject_files/figure-html/unnamed-chunk-11-1.png)<!-- -->
  
Accuracy of 0.99 indicates that in 99% of cases the qualification of execrcise into particular class is correct. Random forest does its job very well and the research of alternative algorithm stops here.

##6 Prediction on validation set##

```r
predictions3 <- predict(modFit2, data_validation, type = "class")
predictions3[1:10]
```

```
##  1  2  3  4  5  6  7  8  9 10 
##  B  A  B  A  A  E  D  B  A  A 
## Levels: A B C D E
```

Good day
