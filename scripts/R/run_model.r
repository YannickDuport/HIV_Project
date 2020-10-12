library(glmnet)
library(HDeconometrics)
library(ggplot2)

# import helper functions
PATH <- dirname(rstudioapi::getSourceEditorContext()$path)
HELPERS_PATH <- paste0(PATH, "/helpers.r")
source(HELPERS_PATH)

# define parameters
FILEPATH <- "~/work/rki/HIV_Project/data/fasta/diversities2.csv"
weights <- rep(1, 105)

# read data containing time and diversity
df <- read.csv(FILEPATH, header=TRUE)

# split data in training and testing sets
l <- split_df(df, 0.7)
df_train <- l$train
df_test <- l$test
x_train <- as.matrix(l$train[, c(-1, -2)])
x_test <- as.matrix(l$test[, c(-1, -2)])
y_train <- l$train$time
y_test <- l$test$time

#################
# Create models #
#################

# define parameters for model selection
lambdas <- logspace(-4, 0, 500)
alphas <- c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
#alphas <- c(1)
# lasso (cross validation)
fit.lassoCV <- cv.glmnet(
  x_train, y_train, weights = weights, 
  lambda = lambdas, alpha = 1, 
  intercept = FALSE
  )

model <- fit.lassoCV
plot(model)
lambda <- model$lambda.1se
pred_test <- predict(model, x_test, s=lambda)
pred_train <- predict(model, x_train, s=lambda)
plot_prediction(y_train, pred_train, y_test, pred_test, lambda, "lasso CV")


# elastic net (cross validation)
mean_errors <- list()
df <- data.frame(matrix(ncol=5, nrow=0))
colnames(df) <- c("alpha",
                  "mse_min", 
                  "mse_1se",
                  "lambda_min",
                  "lambda_1se")

i = 1
for(alpha in alphas) {
  fit.enCV <- cv.glmnet(
    x_train, y_train, weights = weights, 
    lambda = lambdas, alpha = alpha, 
    intercept = FALSE
  )
  
  mean_errors[[i]] <-fit.enCV$cvm
  mse_min <- min(fit.enCV$cvm)
  mse_1se <- fit.enCV$cvm[which(fit.enCV$lambda == fit.enCV$lambda.1se)]
  lambda_min <- fit.enCV$lambda.min
  lambda_1se <- fit.enCV$lambda.1se
  df[i, ] <- c(alpha, mse_min, mse_1se, lambda_min, lambda_1se)
  i <- i + 1
}

idx_min <- which.min(df$mse_min)
idx_1se <- which.max(df$lambda_1se)
lambda_min <- df$lambda_min[idx_min]
lambda_1se <- df$lambda_1se[idx_1se]
alpha_min <- alphas[idx_min]
alpha_1se <- alphas[idx_1se]

# plot traces for each alpha
df.enCV <- data.frame(
  "cvm"=unlist(mean_errors), 
  "lambda"=rep(log(fit.enCV$lambda), length(alphas)),
  "alpha"=rep(alphas, each=length(lambdas))
)
df.enCV$alpha <- as.factor(df.enCV$alpha)
ggplot(df.enCV) + geom_line(aes(x=lambda, y=cvm, color=alpha)) 

# create final model
fit.enCV.min <- cv.glmnet(
  x_train, y_train, weights = weights, 
  lambda = lambdas, alpha = alpha_min, 
  intercept = FALSE)
fit.enCV.1se <- cv.glmnet(
  x_train, y_train, weights = weights, 
  lambda = lambdas, alpha = alpha_1se, 
  intercept = FALSE)

pred_test_min <- predict(fit.enCV.min, x_test, s=lambda_min)
pred_train_min <- predict(fit.enCV.min, x_train, s=lambda_min)
pred_test_1se <- predict(fit.enCV.1se, x_test, s=lambda_1se)
pred_train_1se <- predict(fit.enCV.1se, x_train, s=lambda_1se)

plot_prediction(y_train, pred_train_min, y_test, pred_test_min, lambda_min, "elastic net CV (lambda min)")
plot_prediction(y_train, pred_train_1se, y_test, pred_test_1se, lambda_1se, "elastic net CV (lambda 1se)")

# lasso (aic)
fit.lassoAIC <- ic.glmnet(
  x_train, y_train, weights = weights,
  lambda = lambdas, alpha = 1, crit = "aicc",
  intercept = FALSE
)

model <- fit.lassoAIC
lambda <- round(model$lambda, 3)
pred_test <- predict(model, x_test, s=lambda)
pred_train <- predict(model, x_train, s=lambda)
plot_prediction(y_train, pred_train, y_test, pred_test, lambda, "lasso AIC")
  plot(model)


################
# plot results #
################

# make predictions
# model <- fit.lassoCV
model <- fit.lassoAIC
model <- fit.lassoCV


plot(model, xvar="lambda")
plot(fit.enCV, xvar="lambda")




