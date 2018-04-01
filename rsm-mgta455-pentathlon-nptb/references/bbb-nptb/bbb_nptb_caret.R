## Using nnet and caret to analyses the bbb_nptb data
## loading libraries
library(nnet)
library(caret)
library(radiant)

## loading the bbb_nptb data and removing missing values
bbb_nptb_wrk <- readr::read_rds("data/bbb_nptb.rds") %>%
  na.omit()

## response (rvar), explanatory (evar) variables, and the training variable
rvar <- "buyer"
evar <- c("offer", "gender", "last", "total", "child", "youth", "cook", "do_it", "reference", "art", "geog")
int <- c(
  "offer:gender", "offer:last", "offer:total", "offer:child",  "offer:youth", 
  "offer:cook", "offer:do_it", "offer:reference", "offer:art", "offer:geog"
)
lev <- "yes"
training <- rep(0L, nrow(bbb_nptb_wrk))

set.seed(1234)
ind <- createDataPartition(
  bbb_nptb_wrk$offer,
  p = .8
)[[1]]
training[ind] <- 1L

## create a training and a validation (or test) set
## for caret to work with
df_train <- bbb_nptb_wrk[ind, c(rvar, evar)]
df_test <- bbb_nptb_wrk[-ind, c(rvar, evar)]

## setup a tibble to use for evaluation
eval_dat <- tibble::tibble(
  buyer = bbb_nptb_wrk$buyer, 
  training = training
)

## compare to a logistic regression
result <- logistic(df_train,  rvar = rvar,  evar = evar,  lev = lev, int = int)
summary(result)
eval_dat$logit <- predict(result, bbb_nptb_wrk)$Prediction

## using radiant.model::nn 
result <- nn(df_train, rvar = rvar, evar = evar, lev = lev, size = 2, decay = 0.25, seed = 1234)
summary(result)
eval_dat$nn2r <- predict(result, bbb_nptb_wrk)$Prediction

## standardize data for use with the nnet package
df_train_scaled <- scaledf(df_train)

## scale the bbb data using the mean and standard deviation of the
## training sample (see ?radiant.model::scaledf)
bbb_scaled <- bbb_nptb_wrk %>%
  copy_attr(df_train_scaled, c("ms","sds")) %>%
  scaledf(calc = FALSE)

## running an nnet model with 2 nodes in the hidden layer
## will produce the same results as the nn model above
set.seed(1234)
result <- nnet::nnet(
  buyer == lev ~ ., 
  data = df_train_scaled, 
  size = 2, 
  decay = 0.25,
  rang = .1, 
  linout = FALSE, 
  entropy = TRUE, 
  skip = FALSE,
  trace = FALSE, 
  maxit = 10000
)
eval_dat$nn2 <- predict(result, bbb_scaled)[,1]

## Custom function to evaluate model as ModelMetrics is causing a segfault in R
## see http://topepo.github.io/caret/training.html#metrics
auc <- function(data, lev = NULL, model = NULL) {
  c(auc = radiant.model::auc(data$yes, data$obs, "yes"))
}

## using caret with nnet
## use the big grid ...
# grid <- expand.grid(size = 1:6, decay = seq(0, 1, 0.05))
## ... or the small grid as an example
grid <- expand.grid(size = 3, decay = seq(0.05, 0.15, 0.05))
ctrl <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE,
  summaryFunction = auc, 
  verboseIter = TRUE
)

## this can take quite some time, especially with a big grid and model ...
## comment out to avoid running is not needed
# set.seed(1234)
# result <- train(
#   select(df_train_scaled, -1), 
#   df_train_scaled[[rvar]], 
#   method = "nnet",
#   trControl = ctrl, 
#   tuneGrid = grid, 
#   metric = "auc", 
#   rang = .1,
#   skip = FALSE, 
#   linout = FALSE, 
#   trace = FALSE, 
#   maxit = 10000
# )

##  Running through the above code with cv = 10 produces ...
# grid <- expand.grid(size = 1:6, decay = seq(0, 1, 0.05))
# Aggregating results
# Selecting tuning parameters
# Fitting size = 3, decay = 0.05 on full training set

## adding predictions from the best model according to caret
# eval_dat$nnc <- predict(result, bbb_scaled, type = "prob")[[lev]]

## re-running final model so we can control the seed
set.seed(1234)
result <- nnet::nnet(
  buyer == lev ~ ., 
  data = df_train_scaled, 
  size = 3, 
  decay = 0.05,
  rang = .1, 
  linout = FALSE, 
  entropy = TRUE, 
  skip = FALSE,
  trace = FALSE, 
  maxit = 10000
)
eval_dat$nnc <- predict(result, bbb_scaled)[,1]

## get a list of the models to compare
mods <- colnames(eval_dat)[-1:-2]

## evaluate all models using the validation
evalbin(
  eval_dat,
  pred = mods,
  rvar = rvar,
  lev = lev,
  qnt = 50,
  train = "Validation",
  data_filter = "training == 1"
) %>% plot(plots = "gains")

## evaluate the model picked by caret in training and validation
## for evidence of overfitting in the validation (or test) data
evalbin(
  eval_dat,
  pred = "nnc",
  rvar = rvar,
  lev = lev,
  qnt = 50,
  train = "Both",
  data_filter = "training == 1"
) %>% plot(plots = "gains")

## calculate the confusion matrix and various performance
## metrics for all models
confusion(
  eval_dat,
  pred = mods,
  rvar = rvar,
  lev = lev,
  qnt = 50,
  train = "Validation",
  data_filter = "training == 1"
) %>% summary()

## Using caret and nn for a regression of `total` on the other varialbes in
## the selected data
## Note: This is NOT meant to be a meaningful regression, just an example
grid <- expand.grid(size = 1:2, decay = seq(0.25, 0.5, 0.25))
ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

## this will take some time ...
set.seed(1234)
result <- train(
  select_at(df_train_scaled, setdiff(c(rvar, evar), "total")),
  df_train_scaled$total,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = grid,
  linout = TRUE,
  entropy = FALSE,
  skip = FALSE,
  rang = .1,
  trace = FALSE,
  maxit = 10000
)

## re-estimating the final model so we can control the seed
set.seed(1234)
result <- nnet::nnet(
  total ~ ., 
  data = df_train_scaled, 
  size = 1, 
  decay = 0.25,
  rang = .1, 
  linout = TRUE, 
  entropy = FALSE, 
  skip = FALSE,
  trace = FALSE, 
  maxit = 10000
)

## getting predictions back on the same scale
total_mean <- attributes(bbb_scaled)$ms$total
total_sd <- attributes(bbb_scaled)$sds$total
pred <- predict(result, bbb_scaled, type = "raw") * 2 * total_sd + total_mean
head(pred)

## estimating the same model using radiant.model::nn
result <- nn(
  bbb_nptb_wrk,
  rvar = "total",
  evar = setdiff(c(rvar, evar), "total"),
  type = "regression",
  size = 1,
  decay = 0.25,
  seed = 1234,
  data_filter = "training == 1"
)
summary(result, prn = TRUE)
pred <- predict(result, bbb_nptb_wrk)$Prediction
head(pred)
