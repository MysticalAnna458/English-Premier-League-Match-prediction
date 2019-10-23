#######################################################################################

############################# Libraries Used  #################################
## If any package is not present on your system kindly uncomment to install

#install.packages("readr")
library(readr)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("nnet")
library(nnet)
#install.packages("MASS")
library(MASS)

#if you are using Mac, you need to do "brew install libomp" in the terminal to get the library working 
#install.packages("caret")
library(caret)

#the package DataEcplorer is availabe for R>3.4 
#install.packages("DataExplorer")
library(DataExplorer)

#install.packages("corrplot")
library(corrplot)

################### Data Preprocessing #####################################

#Getting the data into the environment 
match_data = read.csv(file.choose(),header=T)

#Look into the properties of the attributes 
str(match_data)

#Identify Missing Values 
is.na(match_data)

#Remove unwanted columns from the data 
match_data_clean = match_data[,c(1:23)]
str(match_data_clean)
match_data_clean = match_data_clean[,-c(1,2,10,11)]
str(match_data_clean)
match_data_cleanest = match_data_clean[,-c(1,2,3,4)]
str(match_data_cleanest)

################## Performing the EDA ############################################

# summary of the required dataset
summary(match_data_cleanest)

# studying the structure of the dataset 

# Plotting the structue of dataset
plot_str(match_data_cleanest)

# introducing the dataset
introduce(match_data_cleanest)

# plotting the introduced dataset
plot_intro(match_data_cleanest)

# plotting the missing values
plot_missing(match_data_cleanest)

# plotting bar plot of the final dataset
plot_bar(match_data_cleanest)

# Full Time Home goals
ggplot(match_data_clean,aes(x=HomeTeam,y=FTHG))+geom_histogram(stat="identity")

# Full Time Away goals
ggplot(match_data_clean,aes(x=AwayTeam,y=FTAG))+geom_histogram(stat="identity")

# Correlation Martix
correlation<-cor(match_data_cleanest[,c(2,3,4,5,6,7,8,9,10,11,12,13,14,15)])
correlation

# Plotting correlation matrix
Plot_cor<- corrplot(correlation, method="number")

################## Defining test and train dataset ############################################

# splitting the data
set.seed(1000)
index = sample(2,nrow(match_data_cleanest),replace=T,prob=c(0.75,0.25))
train = match_data_cleanest[index==1,]
test = match_data_cleanest[index==2,]

################## Performing the Multinomial Logistic Regression ############################################

set.seed(1000)
model1 = multinom(FTR~HTHG + HTAG + HS + AS + HST + AST + HF + AF + HC + AC + HY + AY + HR + AR, data = train)

#Summary of first Model 
summary(model1)

#Results as probabilties 
head(fitted(model1))

#Intercepts 
exp(coef(model1))

#Check the head of the probability table 
head(probability.table <- fitted(model1))

# Predicting the values for train dataset
train$precticed <- predict(model1, newdata = train, "class")
train$precticed

# Building classification table
ctable <- table(train$FTR, train$precticed)

#Check confusion Matrix 
ctable

# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable))/sum(ctable))*100,2)


# Predicting the values for test dataset
test$precticed <- predict(model1, newdata = test, "class")

# Building classification table
ctable <- table(test$FTR, test$precticed)

# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable))/sum(ctable))*100,2)


#Use AIC model with multinomial logistical regresssion to increase the accuracys
stepAIC(model1,direction="both")

# After AIC model
aicmodel = multinom(FTR~HTHG + HTAG + HST + AST + HF + HC, data = train)
summary(aicmodel)
head(fitted(aicmodel))

# Predicting the values for train dataset
train$precticed1 <- predict(aicmodel, newdata = train, "class")
train$precticed1

# Building classification table
ctable1 <- table(train$FTR, train$precticed1)

# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable1))/sum(ctable1))*100,2)

# Predicting the values for test dataset
test$precticed1 <- predict(aicmodel, newdata = test, "class")

# Building classification table
ctable2 <- table(test$FTR, test$precticed1)

#Check the new COnfusion Matrix 
ctable2

# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable2))/sum(ctable2))*100,2)


################################### Performing the Random Forest ############################################

install.packages('caret', dependencies = TRUE)
library(caret)
install.packages("randomForest")
library(randomForest)

## random forest
rf <- randomForest((FTR) ~., data = train, ntree = 300, mtry = 3, maxnodes= NULL, importance = TRUE)
## variable importance plot
varImpPlot(rf, type = 1)
## confusion matrix
rf.pred <- predict(rf, test)
confusionMatrix(rf.pred, test$FTR)

############################ Accuracy was 66.67 ########################


trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid")
#using the K fold cross validation through the function traincontrol()
set.seed(1000)
# Run the model
rf_default <- train(FTR~.,
                    data = train,
                    method = "rf",
                    metric = "Accuracy",
                    trControl = trControl)

rf_default
#checking the results for the created model with random selection
#The final value used for the model was mtry = 2 with an accuracy of 0.56

#now we will check for best value of mtry

set.seed(1000)
#set the seed for the upcomming code
tuneGrid <- expand.grid(.mtry = c(1: 14))
#select the grid values from 1-14
rf_mtry <- train(FTR~.,
                 data = train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 300)
#build the model for specific values of mtry
rf_mtry

#check the accuracy
####################### Accuracy was highest at mtry=1, accuracy=0.597 #############################


best_mtry <- rf_mtry$bestTune$mtry 
#store the best value of mtry for using it later

#now we will search for the best maxnodes
store_maxnode <- list()
#store the maxnode in the list()
tuneGrid <- expand.grid(.mtry = best_mtry)
#using the best mtry value for tuning the grid
for (maxnodes in c(5: 25)) {
  #selected the maxmodes from 5-25
  set.seed(1234)
  #set seet for reproducible results
  rf_maxnode <- train(FTR~.,
                      data = train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
#check for the maximum accuracy for Maxnodes
results_mtry <- resamples(store_maxnode)
#store the resluts into variable resluts_mtry
summary(results_mtry)
####################### highest accuracy for maxnode 21 which is 0.689 #############################

#trying to check for other max node ranges from 25-40
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
#tuning the grid for maxnodes values range from 25-40
for (maxnodes in c(25: 40)) {
  set.seed(123)
  #set the seed 123 for reproducible results
  rf_maxnode <- train(FTR~.,
                      data = train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  key <- toString(maxnodes)
  store_maxnode[[key]] <- rf_maxnode
}
#checked the maxnode from range 25-40
results_node <- resamples(store_maxnode)
#store the results in variable results_node
summary(results_node)

##################### Key Intakes ####################################

#here as we can see the the max accuracy was 0.703 only

#so we can conclude that highest accuracy was when maxnode was 25

#now we have to search for the best ntrees

store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(1000)
  rf_maxtrees <- train(FTR~.,
                       data = train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 25,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
#check the max value for number of tree ie. ntree
results_tree <- resamples(store_maxtrees)
#store the results in variable result_tree
summary(results_tree)
#ntree=2000 has the highest accuracy of 0.714

#now we have our final model. You can train the random forest with the following parameters

set.seed(1000)
fit_rf <- train(FTR~.,
                train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 2000,
                maxnodes = 25)

#fitting the model by selecting the tuned values

predict(fit_rf, newdata= test)
#doing the prediction on test dataset
prediction1 <-predict(fit_rf, test)
#storing the predicting values to the variable prediction1
confusionMatrix(prediction1, test$FTR)
# final Accuracy =63.64%


################################# Performing the LDA ############################

library(MASS)
linear = lda(FTR~., data=train)
linear
linear$counts

# prediction and calculating accuracy
p = predict(linear,train)
p$posterior
table1 =table(train$FTR,p$class)
table1
accuracy = sum(diag(table1))/sum(table1)
accuracy

# plotting overlap histograms
ldahist(data=p$x[,1],g= train$FTR)
ldahist(data=p$x[,2],g=train$FTR)

#Bi-plot
install.packages("devtools")
library(devtools)
install_github("fawda123/ggord")
library(ggord)
ggord(linear,train$FTR)

################################# Performing the Neural Network ############################

install.packages("nnet")
library(nnet)
install.packages("NeuralNetTools")
library(NeuralNetTools)

train[-1] = scale(train[-1])
test[-1] = scale(test[-1])

# Neural NET 
set.seed(1000)
nn = nnet(FTR~., data=train, size = 5, entropy=T)
plotnet(nn, cex_val =.8,max_sp=T,circle_cex=5,circle_col = 'red')
summary(nn)

# Prediction with NEural NET
pred_nn = predict(nn,test, type="class")
str(pred_nn)

table3 = table(test$FTR, pred_nn)
table3

accuracy2 = sum(diag(table3))/sum(table3)
accuracy2


