# Coursera ML project - Human Activity Recognition
Su Bardo  
November 24, 2016  



## About project
   Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively.
These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their 
health, to find patterns in their behavior, or because they are tech geeks. One 
thing that people regularly do is quantify how much of a particular activity they
do, but they rarely quantify how well they do it. Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front
(Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.
   In this project, our goal will be to use train data set from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to fit the model. After we test our model on the data set which are unseen by our model.
   
## Downloading Libraries & Data
We are preparing data for processing by downloading it from the source site.

```r
setwd("C:/Users/Sumbat/datasciencecoursera/practicalmachinelearning")

library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(parallel)
library(doParallel)

Url4train <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Url4test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
pmltraining<- "pml-training.csv"
pmltesting<- "pml-testing.csv"

if (!file.exists(pmltraining)) {
    download.file(Url4train, destfile=pmltraining)
}
if (!file.exists(pmltesting)) {
    download.file(Url4test, destfile=pmltesting)
}
```

## Reading & Preprocessing Data
Now we read data in train adn test data set.


```r
train<- read.csv(pmltraining, na.strings = c("", NA))
test<- read.csv(pmltesting, na.strings = c("", NA))
summary(train)
```
From **summary** command's output we see there a lot of NA data in train set. Let's 
check how many NA's in columns.

```r
table(colSums(is.na(train)))
```

```
## 
##     0 19216 
##    60   100
```

```r
table(colSums(is.na(test)))
```

```
## 
##   0  20 
##  60 100
```
There about one hundred columns with almost 98 percent of NA in each column in train dataset (the same as to test dataset). So we get rid of those columns with huge amounts of NA's. We also remove columns from 1 to 7 as low informative for our task.

```r
train<- train[, colSums(is.na(train))==0]
test<- test[, colSums(is.na(test))==0]
train<- train[, -(1:7)]
test<- test[, -(1:7)]
```


## Splitting the Train Data 
To conduct cross validation we split train dataset into training (70%) and validation
(30%) data set.

```r
set.seed(1234)
inTrain <- createDataPartition(train$classe, p=0.70, list=F)
trainRF <- train[inTrain, ]
validatRF <- train[-inTrain, ]
```
## Fitting the model
We use Random Forest algorithm. But at the beginning we prepare our ayatem to execite in paralel mode. Then we use 5-fold cross validation.


```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

x<- trainRF[, -53]
y<- trainRF[,  53]

fit.RF <- train(x, y, data=train, method="rf", trControl=fitControl, ntree=250)
```

A while after that (15 minutes fo waiting) we get our model fitted. Now we test our model on validation set.

```r
predict.RF<- predict(fit.RF, validatRF)
confusionMatrix(validatRF$classe, predict.RF)$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9945624      0.9931207      0.9923324      0.9962778      0.2863212 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```
The accuracy of the model on validation set is 99.42%. 

## Predicting outcome
Let's try our model on unseen test dataset.  Before predicting on test dataset problem_id column will be deleted.


```r
test<- test[,-53]
final_prediction <- predict(fit.RF, test)
```
