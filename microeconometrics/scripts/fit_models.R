library(tidyverse)

# Read ols data with first column as index
data_pooled <- read_csv("data/data_ols.csv")[,-1]
# Drop columns country_AUT and ski_fischer
data_pooled = data_pooled %>% select(-c("country_AUT", "skis_fischer"))

# Remove first two columns
data_panel <- read_csv("data/data_panel.csv")[,-c(1,2)]


# Fit pooled ols model
pooled_ols <- lm(data_pooled, formula = z_score ~ .)
summary(pooled_ols)

# Fit panel ols model
panel_ols <- lm(data_panel, formula = z_score ~ .)
summary(panel_ols)
