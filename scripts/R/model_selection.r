library(glmnet)
library(ggplot2)

PATH <- dirname(rstudioapi::getSourceEditorContext()$path)
HELPERS_PATH <- paste0(PATH, "/helpers.r")
source(HELPERS_PATH)


create_model <- function(x, y, weights, model_selection, kwargs) {
  if (model_selection == "lassoCV") {
    model = lasso(x, y, weights)
  }
}

lassoCV <- function(x, y, weights, lambda=NULL) {
  fit <- cv.glmnet(x, y, weights = weights, lambda = lambda, alpha=1
                   )
}

test <- function(x, y) {
  return(x + y)
}