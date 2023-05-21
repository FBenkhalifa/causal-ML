library(tidyverse)
library(ggplot2)
library(ggstatsplot)
library(ggpubr)
library(ggthemes)
library(afex)

data <- read_csv('data/data_preprocessed.csv')
# Add column 'rank' with the rank according to total_wcp_nat_last for each country compared to all other countries
rank_tbl <- data %>%
  group_by(country) %>%
  summarise(median_wcp = median(total_wcp_nat_last),
            n_country = n()) %>%

  mutate(rank_country = rank(-median_wcp, ties.method = 'min')) %>%
  ungroup() %>%
  select(-median_wcp)

# Add column 'rank' with the rank according to total_wcp_nat_last for each country compared to all other countries
data <- data %>%
  left_join(rank_tbl, by = 'country')

# Get the difference between the z_score when treatm
data_differences <- data %>%
  group_by(name) %>%
  summarise(treatment_true = mean(z_score[treatment == TRUE]),
            treatment_false = mean(z_score[treatment == FALSE]),
            z_score_diff = treatment_true - treatment_false
  ) %>%
  inner_join(., data %>% select(country, gender, name, rank_country, n_country, details_competition_type), by = 'name', multiple = 'first')

# Bring in right format for plotting
data_plot <- data_differences %>%
             filter(!is.na(z_score_diff)) %>%
             pivot_longer(
               cols = c('treatment_true', 'treatment_false'),
               names_to = 'treatment',
               values_to = 'z_score') %>%
               mutate(treatment = ifelse(treatment == 'treatment_true', 'treated', 'not treated'))


data_plot <- data_plot %>% mutate(gender_type = paste0(details_competition_type, ' - ', gender ))

# Order gender_type factor type by alphabetical order
data_plot$gender_type <-
  factor(data_plot$gender_type,
         levels = c("Super G - Female",
                    "Slalom - Female",
                    "Giant Slalom - Female",
                    "Super G - Male",
                    "Slalom - Male",
                    "Giant Slalom - Male"
))

plot <- grouped_ggwithinstats(
  data=data_plot,
  x = treatment,
  y = z_score,
  xlab="Treatment",
  ylab="Z-standardized (per race) run time",
  grouping.var = gender_type,
  type= "nonparametric",
  p.adjust.method = "bonferroni",
  # pairwise.display = "significant",
  plotgrid.args = list(nrow = 2),
                       )

ggsave(plot, filename = 'plots/z_score_boxplot.pdf', width = 30, height = 20, units = 'cm')

