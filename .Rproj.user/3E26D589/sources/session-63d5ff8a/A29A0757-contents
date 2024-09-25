library(readr)
library(tidymodels)
library(usemodels)
### Data
honey <- read_csv("Honey.csv")

# Explore the data
honey
unique(honey$Pollen_analysis)

str(honey)
summary(honey)
skimr::skim(honey)

honey |> 
  group_by(Pollen_analysis) |> 
  count()

# We will filter the first three Pollen to reduce the computing time since the data is very large
honey <- honey |>
  select(-Price) |> # drop the price variable
  filter(Pollen_analysis %in% c("Acacia","Alfalfa","Avocado"))

############# Check the distribution of the target variable

ggplot(honey, aes(x = Purity, y=after_stat(density))) + 
  geom_histogram() +
  geom_density()

# It is obvious the target variable is not normally distributed. This a signal
# that we have to be careful how we partition the data into training and test sets

###################### Correlation of the variables
honey |> 
  select(-Pollen_analysis) |>
  correlations() |> round(digits = 2)



# Set the random number stream using `set.seed()` so that the results can be 
# reproduced later. 
# split a balanced data (80/20)
# The data split is done using stratified sampling
set.seed(202)
honey_split <- initial_split(honey, prop = 0.80, strata = Purity)
honey_train <- training(honey_split)
honey_test  <-  testing(honey_split)

dim(honey_train)
dim(honey_test)
# The function `strata` divides the strata variable by four (by default) and 
# samples within each stratum. The default number (four) can be changed using
# the `breaks` function.The change is necessary if the distribution of the test
# set is different from the training set


#################### Perform histogram with the split

train <- ggplot(honey_train, aes(x = Purity, y=after_stat(density))) + 
  geom_histogram() +
  geom_density()+
  labs(title = "Train")

test <- ggplot(honey_test, aes(x = Purity, y=after_stat(density))) + 
  geom_histogram() +
  geom_density()+
  labs(title = "Test")

gridExtra::grid.arrange(train, test, ncol=2)


# re-sampling the training set using 10 fold CV (stratified sampling)
set.seed(200)

honey_CV <- vfold_cv(honey_train, v = 10, 
                     strata = Purity, repeats = 1)
honey_CV
# For a more stable and reliable model, repeated cross validation is recommended.
# For the number of times to repeat the k-fold, there is no one size fit it all.
# Up to 10 repeats is usually recommended for a stable model.
# This configuration is used in order to reduce computing time.

# Preprocessing the data using the `recipe` function
# The follow steps are necessary for this data
honey_recipe <- 
  recipe(Purity ~., data = honey_train) |> 
  step_scale(all_numeric_predictors()) |>  
  step_center(all_numeric_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_zv(all_predictors())
# `step_zv`Deletes any zero-variance predictors that have a single unique value.
# In case there is a factor level that was never observed in the training data 
# (resulting in a column of all 0s).

summary(honey_recipe)

####### Specifying the model
### STEPS
# Choose a model
# Specify an engine
# Set the mode

# Some regression algorithms present in tidymodels algorithms are random forest,
# ordinary least squares, Linear support vector machines (SVMs),Boosted trees, 
# etc. We are going to demonstrate with the random forest algorithm 
####### Specify Engine
  rf_model <- rand_forest(trees = 2000, min_n = tune(),mtry = tune()) |>  
    set_engine("ranger", verbose = TRUE) |>  
    set_mode("regression")

# The hyperparameters in random forest are `trees`,`mtry` and `min_n`

# Setting the workflow  
rf_wflow <- 
  workflow() |>  
  add_model(rf_model) |>  
  add_recipe(honey_recipe) 


# Setting evaluation metrics
# Set evaluation metrics for regression
# Root Mean square Error (RMSE)
# R-Square
# Mean Absolute Error (MAE)
rf.reg_metric <- metric_set(rmse, rsq, mae)

# Tune hyperparameters
rf_grid <- grid_regular(
  mtry(range = c(1, 5)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Control aspects of the grid search process
ctrl <- control_resamples(save_pred = TRUE, verbose = TRUE)

# Search grid
set.seed(203)
start.time <- Sys.time()
honey_rf_model <- 
  tune_grid(
    rf_wflow,
    resamples = honey_CV,
    control = ctrl, 
    metrics = rf.reg_metric
  )
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

# The best hyperparameter configuration using `rmse` is 2000 tress, 7 `mtry`, and 10 `min_n`
# These parameters will be used to fit the final model.
# The run run time is 2.743102 hours for 10 fold repeated once. Therefore, it 
# would take about 27.43102 hours to repeat the 10 fold ten times on my machine.
# 32GBRAM, 1.5GB SSD, 2.1GHz Core i7

# View results
collect_metrics(honey_rf_model)

collect_predictions(honey_rf_model)

# Find the best hyperparameter configuration to fit the final model
honey_rf_model |>    
  show_best(metric = "rsq") 

honey_rf_model |> 
  select_best(metric = "rsq")

honey_rf_model |> 
  select_best(metric = "mae")

best_params <- honey_rf_model |> 
  select_best(metric = "rmse")

# The best hyperparameters configuration based on rmse is 2000 trees,
# 7 mtry and 10 min_n. These hyperparameters are used to fit the final model.

# Finalising the model
# Work flow method
final_workflow <- rf_wflow |>  
  finalize_workflow(best_params)

final_workflow

set.seed(204)


# Train the final model with the best hyperparameters
final_model <- fit(final_workflow, data = honey_train)

# Parameter estimate - Train set
augment(final_model, honey_train) |> 
  rf.reg_metric(truth = Purity, estimate = .pred)
# rmse is 0.0110
# r squared is 0.994
# mae is 0.00288



# Predict with test set

set.seed(205)
result <- final_workflow |> 
  last_fit(honey_split, metrics = rf.reg_metric)

result
 # Note that honest_split contains both training and test sets

# plot the actual and the predicted
library(probably)
result |>  
  collect_predictions() |>  
  cal_plot_regression(
    truth = Purity, 
    estimate = .pred)

# Collect the evaluation metrics
result |> collect_metrics()

# Collect the predictions
mod_pred <- collect_predictions(result)

# Alternatively
# Extract model to make prediction with the test set
pred <- augment(final_model, honey_test)

augment(final_model, honey_test) |> 
  rf.reg_metric(truth = Purity, estimate = .pred)
# rmse is 0.0197
# r squared is 0.980
# mae is 0.00536




