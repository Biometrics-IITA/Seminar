library(readr)
library(tidymodels)
library(usemodels)

### Import Data
apple <- read_csv("Apple.csv")

apple

# The `A_id` is not needed and will be dropped from the dataframe

apple <- apple |> 
  select(-A_id)
# Explore the data
unique(apple$Quality)

str(apple)
summary(apple)
skimr::skim(apple)

apple |> 
  group_by(Quality) |> 
  count()

# From the summary, we see that the target variable is a character variable
# We have to convert it to factor since it is the type required in the prediction 
# model. Also, our interest is predict the good apple, so we have to change the 
# levels and make `good` the first level.

apple <- apple |> 
  mutate(Quality = factor(Quality, levels = c("good","bad")))

levels(apple$Quality)

############# Check the distribution of the target variable

ggplot(apple, aes(x = Quality, fill = Quality))+ 
  geom_bar()+ 
  guides(fill = "none")

# The two classes of the target variable are evenly distributed

################## Correlation of the variables

apple |> 
  psych::pairs.panels(gap = 0,pch=21)

# There is no evidence of multicolinearity among the predictors since there is no 
# strong correlations between any two variable. Additionally, all the features
# are normally distributed as shown in the plot, therefore, no scaling is required.


# Set the random number stream using `set.seed()` so that the results can be 
# reproduced later. 
# split a balanced data (80/20)
# The data split is done using stratified sampling
set.seed(202)
apple_split <- initial_split(apple, prop = 0.80, strata = Quality)
apple_train <- training(apple_split)
apple_test  <-  testing(apple_split)

dim(apple_train)
dim(apple_test)
# The function `strata` divides the strata variable by four (by default) and 
# samples within each stratum. The default number (four) can be changed using
# the `breaks` function.The change is necessary if the distribution of the test
# set is different from the training set


# re-sampling the training set using 10 fold CV (stratified sampling)
set.seed(200)
apple_CV <- vfold_cv(apple_train, v = 10, 
                     strata = Quality, repeats = 5)
apple_CV

# Preprocessing the data using the `recipe` function

apple_recipe <- 
  recipe(Quality ~., data = apple_train) # specify the formula

summary(apple_recipe)

####### Specifying the model
### STEPS
# Choose a model
# Specify an engine
# Set the mode


####### Specify Engine
rf_model <- rand_forest(trees = 2000, min_n = tune(),mtry = tune()) |>  
  set_engine("ranger", verbose = TRUE) |>  
  set_mode("classification")
# The hyperparameters for random forest are the number of trees (`tress`)
# `mtry`: The number of features to randomly select select at each split
# `min_n`: A minimum number of data points in a node that are required for 
# the node to be split further

# Setting the workflow
# A workflow is a container object that aggregates information required to fit 
# and predict from a model
rf_wflow <- 
  workflow() |>  
  add_model(rf_model) |> 
  add_recipe(apple_recipe) 

# Setting evaluation metrics
# `accuracy`: the proportion of the data that are predicted correctly
# `roc_auc`: Area Under the Receiver Operating Characteristic Curve
# pr_auc: Area Under the Precision-Recall Curve 
rf.reg_metric <- metric_set(accuracy,roc_auc, pr_auc)

# Tune hyperparameters
rf_grid <- grid_regular(
  mtry(range = c(3, 7)),
  min_n(range = c(5, 20)),
  levels = 5
)

# Control aspects of the grid search process
ctrl <- control_resamples(save_pred = TRUE, verbose = TRUE)

# Search grid
set.seed(203)
start.time <- Sys.time()
apple_rf_model <- 
  tune_grid(
    rf_wflow,
    resamples = apple_CV,
    control = ctrl, 
    metrics = rf.reg_metric
  )
end.time <- Sys.time()
run.time <- end.time - start.time
run.time

# The run time is 2.135667 hours

# View results
collect_metrics(apple_rf_model)

collect_predictions(apple_rf_model)

# Find the best hyperparameter configuration  
apple_rf_model |>    
  show_best(metric = "accuracy") 

# The best hyperparameters configuration here based on accuracy is trees=2000,
# mtry =5 and min_n=10. 
best_params <- apple_rf_model |> 
  select_best(metric = "accuracy")

# Final model
final_workflow <- rf_wflow |>  
  finalize_workflow(best_params)

final_workflow

final_model <- fit(final_workflow, data = apple_train,)

# Performance of the training set
# Confusion matrix
# A confusion matrix is a table that is used to define the performance of a 
# classification algorithm. 

augment(final_model, apple_train) |> 
  conf_mat(truth = Quality, estimate = .pred_class) |> 
  autoplot(type = "heatmap")

# The confusion matrix shows that 6 good apples are wrongly classified as bad and 
# 7 bad apples are wrongly classified as good.

# setting the metrics
metrix <- metric_set(accuracy, sensitivity, specificity)
# Accuracy is the proportion of all classifications that were correct, whether 
# positive or negative

# Sensitivity (recall) measures the proportion of actual positives that are correctly 
# identified by the model as positive. TP/(TP+FN)

# Specificity measures the proportion of actual negatives that are correctly 
# identified as negative TN/(TN+FP)


augment(final_model, apple_train) |> 
  metrix(truth = Quality, estimate = .pred_class)
# The accuracy of the final model is 99.6%
# sensitivity is 99.6%
# specificity is 99.6%

# Test set
# Using a workflow method
set.seed(205)
result <- final_workflow |> 
  last_fit(apple_split, metrics = rf.reg_metric)

result



result |> collect_metrics()

mod_pred <- collect_predictions(result)

augment(final_model, apple_test) |> 
  metrix(truth = Quality, estimate = .pred_class)
# For the test set,the accuracy is 88.8%, sensitivity is 91.8% and 
# specificity is 85.8% 

augment(final_model, apple_test) |> 
  conf_mat(truth = Quality, estimate = .pred_class) |> 
  autoplot(type = "heatmap")

# The confusion matrix shows that 33 bad apples are wrongly classified as
# good and 57 good are wrongly classified as bad

save.image("Classification.RData")

