

library(ggplot2) 
library(readr)
library(caTools)
library(corrplot)
library(caret)
library(plyr)
library(randomForest)
library(MLmetrics)


df <- read.csv("/Users/salil/Downloads/train_house.csv",header = TRUE)
test_kaggle <- read.csv("/Users/salil/Downloads/test_house.csv",header = TRUE)
# str(df)
# colSums(is.na(df))

# Data Pre-processing
# We can see that the NA values do not mean the data has empty/unavailable values ,but the particular features do not have the specific data.
# We are replacing such NAs with "None" for categorical variables and "0" for Numeric.
# Removing features PoolQC and MiscFeature since they have more than 90% NA values.

df$PoolQC<-df$MiscFeature<-NULL
df_na <- df[colSums(is.na(df)) > 0]
na_cols <- colnames(df_na)

imp_df <- df_na[,c(1,4,13,2,3,5:12,14:17)]
#summary(imp_df)
imp_df_cat <- imp_df[4:17]
imp_df_num <- imp_df[1:3]

# Replacing categorical values with None
imp_df_cat_mat <- as.matrix(imp_df_cat)     # convert to matrix 
y <- which(is.na(imp_df_cat)==TRUE)         # get index of NA values 
imp_df_cat_mat[y] <- "None" 
imp_df_cat <- data.frame(imp_df_cat_mat)

# Replacing numerical values with 0
imp_df_num_mat <- as.matrix(imp_df_num)     # convert to matrix 
x <- which(is.na(imp_df_num)==TRUE)         # get index of NA values 
imp_df_num_mat[x] <- 0
imp_df_num <- data.frame(imp_df_num_mat)

# Merging the imputed values back to the original data set
imputed <- cbind(imp_df_num, imp_df_cat)
rem_cols <- setdiff(colnames(df),na_cols)
df_final_data <- cbind(df[rem_cols],imputed)
# summary(df_final)
# dim(df_final)

# Removing outliers with boxplot
bp <- boxplot(df_final_data$SalePrice)
df_final <- df_final_data[!(df_final_data$SalePrice %in% bp$out),]
dim(df_final)

index <- grep("SalePrice", colnames(df_final))
df_final <- cbind(df_final[-index], df_final[index])

# Split the dataset into training,cross validation and test sets.

sample_data <- sample.split(df_final$SalePrice,SplitRatio=0.7)
train_house <- subset(df_final,sample_data == TRUE)
test_house <- subset(df_final,sample_data == FALSE)

# Split the test set further into cross validation and a final test set (15/15)
cv_test <- sample.split(test_house$SalePrice,SplitRatio = 0.5)
cv_house <- subset(test_house,cv_test == TRUE)
testing_house <- subset(test_house,cv_test == FALSE)

# Feature extraction for the given model using Recursive Feature Extraction by Random Forest 

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
result <- rfe(df_final[ ,1:78],df_final[ ,79],rfeControl=control)

# Selecting predictors with significant importance
plot(result, type=c("g", "o"))

#As we can see RMSE is almost the same for the top 16 variables. 
# Hence, including those as the final features to test on cross validation and test sets.
var_final <- head(predictors(result),16)
var_final

# Visualizations & Correlations
# One hot encoding to find corelations between variables
dummy <- dummyVars( ~ . , data = df_final)
ohe <- data.frame(predict(dummy, newdata = df_final))
m <- cor(ohe)
# corrplot(m, method = "number")
ggplot(df_final,aes(SalePrice,YearBuilt,color=OverallQual,size=LotArea)) + geom_point()  + labs(x="SalePrice",y="Year",title="SalePrice vs Year")

# Once we have imputed the data, we will train the data on Linear Regression & Random Forest.
# Model fit on Linear Regression
#fit_lm <- lm (SalePrice ~ GrLivArea + Neighborhood + OverallQual + TotalBsmtSF + X1stFlrSF + X2ndFlrSF + GarageCars + BsmtFinSF1 + GarageArea + ExterQual + LotArea + BsmtFinType1 + FireplaceQu + GarageType + KitchenQual  + YearBuilt , data = train_house)
#pred_cv_lm <- predict(fit_lm,cv_house)
#rmse_cv_lm <- RMSE(pred_cv_lm,cv_house$SalePrice)

#pred_test_lm <-  predict(fit_lm,testing_house)
#rmse_test_lm <- RMSE(pred_test_lm,testing_house$SalePrice)

# Model fit on a Random Forest 
fit_rf <- randomForest (SalePrice ~ GrLivArea + Neighborhood + OverallQual + TotalBsmtSF + X1stFlrSF + X2ndFlrSF + GarageCars + BsmtFinSF1 + GarageArea + ExterQual + LotArea + BsmtFinType1 + FireplaceQu + GarageType + KitchenQual  + YearBuilt , data = train_house , mtry = 6, ntree = 500 , nodesize = 5 )
pred_cv_rf <- predict(fit_rf,cv_house)
rmse_cv_rf <- RMSE(pred_cv_rf,cv_house$SalePrice)

pred_test_rf <-  predict(fit_rf,testing_house)
rmse_test_rf <- RMSE(pred_test_rf,testing_house$SalePrice)

err_df <- data.frame("RMSE" = c(rmse_cv_lm,rmse_cv_rf,rmse_test_lm,rmse_test_rf), "type" = c("lm","rf"))
err_df
# Logarithmic RMSE 
rmse_log_cv_rf <- RMSLE(pred_cv_rf,cv_house$SalePrice)
rmse_log_test_rf <- RMSLE(pred_test_rf,testing_house$SalePrice)
rmse_log_cv_rf
rmse_log_test_rf

param_list <- list(
  objective = 'reg:linear',
  colsample_bytree=0.5,
  subsample=0.5,
  eta=0.2,
  max_depth = 6,
  min_child_weight=3,
  alpha=0.3,
  lambda=3,
  gamma=0.01,
  eval_metrics = 'rmse' 
)

common <- intersect(names(train_house), names(test_kaggle)) 
for (p in common) { 
       if (class(train_house[[p]]) == "factor") { 
        levels(test_kaggle[[p]]) <- levels(train_house[[p]]) 
      } 
}
