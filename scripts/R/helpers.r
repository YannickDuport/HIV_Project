library(ggplot2)

split_df <- function(df, ratio) {
  set.seed(1)
  training_size <- round(nrow(df) * ratio, 0)
  training_idx <- sample(1:nrow(df), training_size, replace=FALSE)
  df_train <- df[training_idx, ]
  df_test <- df[-training_idx, ]
  df_list <- list("train" = df_train, "test" = df_test)
  return(df_list)
}

logspace <- function(x1, x2, n) {
  return(exp(log(10) * seq(x1, x2, length.out = n)))
}

plot_prediction <- function(y_train, pred_train, y_test, pred_test, lambda, title) {
  df <- data.frame(
    "y" = c(y_train, y_test), 
    "prediction" = c(pred_train, pred_test),
    "type" = c(rep("train", length(y_train)), rep("test", length(y_test)))
    )
  mse_train <- mse(y_train, pred_train)
  mse_test <- mse(y_test, pred_test)

  annotation = paste(
    "lambda =", lambda,
    "\nmse (training data) =", round(mse_train, 2),
    "\nmse (testing data) =", round(mse_test, 2),
    "\nmse ratio (test/train)", round(mse_test/mse_train, 3)
  )
  
  p <- ggplot(df) +
    geom_point(aes(y, prediction, color=type)) +
    geom_line(aes(df$y, df$y)) +
    annotate("text", 5, 95, label = annotation, hjust=0) +
    labs(title=title)
  
  return(p)
}

mse <- function(y, y_hat) {
  residuals <- y - y_hat
  mse <- mean(residuals^2)
  return(mse)
}