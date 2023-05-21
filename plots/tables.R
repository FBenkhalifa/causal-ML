library(tidyverse)
library(stargazer)
library('webshot2')
library(plm)
data_ols <- read_csv('data/ols_data.csv') %>% mutate(date = paste0(date, '-', start_time),
                                                     treatment = as.integer(treatment))
data_panel <- read_csv('data/panel_data.csv')



ols <- lm(z_score ~ ., data = data_ols %>% select(-name, -date))
fixed_effects <- lm(z_score ~ ., data = data_panel %>% select(-name, -date))
plr <- lm(z_score ~ ., data = data_panel %>% select(treatment, z_score))

tbl <- stargazer(ols,fixed_effects,plr,
          type = "latex",
          title='Regression results',
          digits = 2,
          report = "vc*",
          single.row = TRUE
)
print(tbl)


# Define the dependent variable
dep_var <- "z_score"
ind_vars <- names(data_ols)[!names(data_ols) %in% c(dep_var, 'name', 'date')]
formula_str <- paste(dep_var, "~", paste(ind_vars, collapse = " + "))
formula <- as.formula(formula_str)

fixed_effects_ols <- plm(formula, data = data_ols, index=c('name', 'date'), model='within', effect='time')
summary(fixed_effects)