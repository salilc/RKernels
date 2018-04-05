# Carvana:Dont get kicked solution

library(ggplot2) 
library(readr)
library(corrplot)
library(caTools)
library(caret)
library(ranger)
library(pROC)

# Reading the dataset

train <- read.csv("/Users/salil/Downloads/training_car.csv")
train_df <- data.frame(train)
# summary(train_df)
# dim(train_df)

#Data pre-processing
#Removing redundant variables that will not improve the outcome of the prediction
# PurchDate and VehYear are the same as VehicleAge
# Removing WheelTypeID as it is closely related to WheelType.
# We can see that AUCGUART & PRIMEUNIT have majority of null values
# Similar redundant variables include Nationality,VNZIP1,VNST,BYRNO

train_df$PurchDate<-train_df$VehYear<-train_df$Color<-train_df$Auction<-train_df$WheelTypeID<-train_df$Nationality<-train_df$VNZIP1<-train_df$VNST<-train_df$BYRNO<-train_df$AUCGUART<-train_df$PRIMEUNIT<-train_df$IsOnlineSale<-NULL
# str(train_df)
# Set NULL values of Trim,Size,WheelType,Transmission as NA

train_df$Trim[grep("NULL", train_df$Trim , ignore.case=TRUE)] <- NAle
train_df$Size[grep("NULL", train_df$Size , ignore.case=TRUE)] <- NA
train_df$TopThreeAmericanName[grep("NULL", train_df$TopThreeAmericanName , ignore.case=TRUE)] <- NA
train_df$WheelType[grep("NULL", train_df$WheelType , ignore.case=TRUE)] <- NA
train_df$Transmission[grep("NULL", train_df$Transmission , ignore.case=TRUE)] <- NA
train_df$MMRCurrentAuctionAveragePrice[grep("NULL", train_df$MMRCurrentAuctionAveragePrice , ignore.case=TRUE)] <- NA
train_df$MMRCurrentAuctionCleanPrice[grep("NULL", train_df$MMRCurrentAuctionCleanPrice , ignore.case=TRUE)] <- NA
train_df$MMRCurrentRetailAveragePrice[grep("NULL", train_df$MMRCurrentRetailAveragePrice , ignore.case=TRUE)] <- NA
train_df$MMRCurrentRetailCleanPrice[grep("NULL", train_df$MMRCurrentRetailCleanPrice , ignore.case=TRUE)] <- NA
# summary(train_df)

# Set value of Manual as MANUAL for Transmission
train_df$Transmission[grep("Manual", train_df$Transmission , ignore.case=FALSE)] <- "MANUAL"
train_df$Transmission <- factor(train_df$Transmission, levels=c("AUTO","MANUAL"))

# Remove NULL as factor levels

train_df<-train_df[ train_df$Size != "NULL", , drop=FALSE]
train_df$Size <- factor(train_df$Size)

train_df<-train_df[ train_df$Trim != "NULL", , drop=FALSE]
train_df$Trim <- factor(train_df$Trim)

train_df<-train_df[ train_df$TopThreeAmericanName != "NULL", , drop=FALSE]
train_df$TopThreeAmericanName <- factor(train_df$TopThreeAmericanName)

train_df<-train_df[ train_df$WheelType != "NULL", , drop=FALSE]
train_df$WheelType <- factor(train_df$WheelType)

train_df<-train_df[ train_df$Transmission != "NULL", , drop=FALSE]
train_df$Transmission <- factor(train_df$Transmission)

#Set MMR prices as integers instead of factors

train_df$MMRAcquisitionAuctionAveragePrice <- as.integer(train_df$MMRAcquisitionAuctionAveragePrice)
train_df$MMRAcquisitionAuctionCleanPrice <- as.integer(train_df$MMRAcquisitionAuctionCleanPrice)
train_df$MMRAcquisitionRetailAveragePrice <- as.integer(train_df$MMRAcquisitionRetailAveragePrice)
train_df$MMRAcquisitonRetailCleanPrice <- as.integer(train_df$MMRAcquisitonRetailCleanPrice)
train_df$MMRCurrentAuctionAveragePrice <- as.integer(train_df$MMRCurrentAuctionAveragePrice)
train_df$MMRCurrentAuctionCleanPrice <- as.integer(train_df$MMRCurrentAuctionCleanPrice)
train_df$MMRCurrentRetailAveragePrice <- as.integer(train_df$MMRCurrentRetailAveragePrice) 
train_df$MMRCurrentRetailCleanPrice <- as.integer(train_df$MMRCurrentRetailCleanPrice)

# Handling NA values in the data set
train_df_final <- train_df[complete.cases(train_df),]
# summary(train_df_final)
# dim(train_df_final)

#Some visualizations with the predictor variables
ggplot(train_df_final,aes(VehicleAge,VehBCost,color=Make)) + geom_point()  + labs(x="Age",y="Cost",title="Vehicle Age vs Cost")
ggplot(train_df_final,aes(MMRCurrentAuctionAveragePrice,MMRAcquisitionAuctionAveragePrice,color=WheelType,size=WarrantyCost)) + geom_point()  + labs(x="Current",y="Acquisition",title="Auction Average vs Current Average price")

# Correlation matrix for the dataset
cor_cols <- c("VehicleAge","VehOdo","MMRAcquisitionAuctionAveragePrice","MMRAcquisitionAuctionCleanPrice","MMRAcquisitionRetailAveragePrice","MMRAcquisitonRetailCleanPrice","MMRCurrentAuctionAveragePrice","MMRCurrentAuctionCleanPrice","MMRCurrentRetailAveragePrice","MMRCurrentRetailCleanPrice","VehBCost","WarrantyCost")
cor_car <- cor(train_df_final[,cor_cols])
corrplot(cor_car,method = "number")

# Split the data into training,cross-validation and test sets (60/40)
sample_data <- sample.split(train_df_final$IsBadBuy,SplitRatio=0.6)
train_car <- subset(train_df_final,sample_data == TRUE)
test_car <- subset(train_df_final,sample_data == FALSE)
# dim(train_car)
# dim(test_car)

# Splitting the test set further into cross validation and a final test set (20/20)
cv_test <- sample.split(test_car$IsBadBuy,SplitRatio = 0.5)
cv_car <- subset(test_car,cv_test == TRUE)
testing_car <- subset(test_car,cv_test == FALSE)

# Training the model using the ranger algorithm which is a faster implementation of Random Forest
set.seed(100)
model <- ranger(IsBadBuy ~ ., data = train_car, num.trees = 500, mtry = 5,importance = "impurity", write.forest = TRUE, min.node.size = 1,verbose = TRUE, replace = FALSE) 

# Plotting variable importance 
df <- data.frame(model$variable.importance)
df <- cbind(rn = rownames(df), df, row.names = NULL)
names(df)[2] <- "importance"
names(df)[1] <- "variable"
ggplot(df,aes(importance,variable)) + geom_point() 

# Predicting on Cross validation set
prob_cv <- predict(model,cv_car)
cv_car$predicted.IsBadBuy <- 0
cv_car[prob_cv$predictions > 0.5, ]$predicted.IsBadBuy <- 1 
# Confusion matrix for Cross-Validation set
conf_cv <- confusionMatrix(table(cv_car$IsBadBuy,cv_car$predicted.IsBadBuy))
conf_cv
# ROC curve for cross validation set
plot(roc(cv_car$IsBadBuy,prob_cv$predictions,direction="<"))

#Predicting on final test set
prob_test <- predict(model,testing_car)
testing_car$predicted.IsBadBuy <- 0
testing_car[prob_test$predictions > 0.5, ]$predicted.IsBadBuy <- 1

# Confusion matrix for test set
conf_test <- confusionMatrix(table(testing_car$IsBadBuy,testing_car$predicted.IsBadBuy))
conf_test

# ROC curve for cross validation set
plot(roc(testing_car$IsBadBuy,prob_test$predictions,direction="<"))