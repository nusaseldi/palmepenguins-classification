# Load library
library(tidyverse)
library(tidymodels)
library(palmerpenguins)
library(vip)

# dataset
penguin_df <- penguins

glimpse(penguin_df)

# viz the dataset
penguin_df <- penguin_df |> 
  drop_na(sex) |> 
  select(-year, -island)

penguin_df |> 
  ggplot(aes(bill_length_mm, bill_depth_mm, color = sex)) +
  geom_point() +
  facet_wrap(~species)

penguin_df |> 
  ggplot(aes(species, body_mass_g , color = sex)) +
  geom_boxplot() 

penguin_df |> 
  ggplot(aes(flipper_length_mm, body_mass_g, color = sex)) +
  geom_point() +
  facet_wrap(~species)

# build a model
set.seed(99)

penguin_split <- initial_split(penguin_df, prop = 0.7, strata = sex)
penguin_train <- training(penguin_split)
penguin_test <- testing(penguin_split)

penguin_fold <- vfold_cv(data = penguin_train, strata = sex)

bt_spec <- boost_tree() |> 
  set_mode('classification') |> 
  set_engine('xgboost')

penguin_recipe <- recipe(sex ~ ., data = penguin_train) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_dummy(all_nominal_predictors())

penguin_wf <- workflow() |> 
  add_recipe(penguin_recipe) |> 
  add_model(bt_spec)

bt_fit <- penguin_wf |> 
  fit_resamples(resamples = penguin_fold,
                control = control_resamples(save_pred = TRUE))

# evaluating the model
collect_metrics(bt_fit)  

penguin_final <- penguin_wf |> 
  last_fit(penguin_split)

collect_metrics(penguin_final)

result <- collect_predictions(penguin_final) 

result |> conf_mat(sex, .pred_class)

penguin_final |> 
  extract_fit_parsnip() |> 
  vip()



