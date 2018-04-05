library(caTools)

user_payroll <- read.csv("/Users/salil/Downloads/UserPayrollTracking.csv")
bank_transact <- read.csv("/Users/salil/Downloads/BankTransactions_credit.csv")
user_payroll$IsPayroll <- 0
user_payroll[user_payroll$TransactionId != "", ]$IsPayroll <- 1
df <- merge(user_payroll[,c("UserId","TransactionId","IsPayroll")],bank_transact,by= c("UserId","TransactionId"),all.y = TRUE)
df["IsPayroll"][is.na(df["IsPayroll"])] <- 0

sample_data <- sample.split(df$IsPayroll,SplitRatio=0.6)
training <- subset(df,sample_data == TRUE)
test_data <- subset(df,sample_data == FALSE)
cv_test <- sample.split(test_data$IsPayroll,SplitRatio = 0.5)
cv <- subset(test_data,cv_test == TRUE)
testing <- subset(test_data,cv_test == FALSE)

# Training and Prediction using rpart
model <- rpart(IsPayroll ~ ., data = training,method = "class")
pred_cv <- predict(model,cv)
pred_test <- predict(model,testing)


confusionMatrix(table(cv$IsPayroll,pred_cv[,2]),positive = "1")
precision(table(cv$IsPayroll,pred_cv[,2]))
recall(table(cv$IsPayroll,pred_cv[,2]))
F_meas(table(cv$IsPayroll,pred_cv[,2]))

confusionMatrix(table(testing$IsPayroll,pred_test[,2]),positive = "1")
precision(table(testing$IsPayroll,pred_test[,2]))
recall(table(testing$IsPayroll,pred_test[,2]))
F_meas(table(testing$IsPayroll,pred_test[,2]))

# Training and Prediction using ranger
fit <- ranger(IsPayroll ~ ., data = training,num.trees = 500,mtry = 3,min.node.size = 1,write.forest = TRUE, verbose = TRUE)
pred_cv_rf <- predict(fit,cv)
pred_test_rf <- predict(fit,testing)

confusionMatrix(table(cv$IsPayroll,as.integer(pred_cv_rf$predictions)),positive = "1")
precision(table(cv$IsPayroll,as.integer(pred_cv_rf$predictions)))
recall(table(cv$IsPayroll,as.integer(pred_cv_rf$predictions)))
F_meas(table(cv$IsPayroll,as.integer(pred_cv_rf$predictions)))


confusionMatrix(table(testing$IsPayroll,as.integer(pred_test_rf$predictions)),positive = "1")
precision(table(testing$IsPayroll,as.integer(pred_test_rf$predictions)))
recall(table(testing$IsPayroll,as.integer(pred_test_rf$predictions)))
F_meas(table(testing$IsPayroll,as.integer(pred_test_rf$predictions)))
