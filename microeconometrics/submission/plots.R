library(tidyverse)
library(ggplot2)
library(ggstatsplot)
library(ggpubr)
library(ggthemes)
library(afex)
library(tidyverse)
library(hrbrthemes)
library(viridis)
library(quantreg)

data <- read_csv('data/data_plot.csv')

data <- data %>%
  group_by(race_id) %>%
  mutate(mean_run_time = mean(run_time, na.rm = TRUE)) %>%
  ungroup()

# Calculate mean_run_time for each race_id
mean_run_times <- data %>%
  group_by(race_id) %>%
  summarise(mean_run_time = mean(run_time, na.rm = TRUE))

# Calculate the quantiles
quantiles <- rq(mean_run_time ~ 1, tau = seq(0, 1, by = 0.1), data = mean_run_times)

# Get the race_ids at each quantile
selected_ids <- sapply(quantiles$coefficients, function(x) {
  which.min(abs(mean_run_times$mean_run_time - x))
})
ids <- data[selected_ids, 'race_id'] %>% pull()
# Filter the original data tibble
filtered_data <- data %>%
  filter(race_id %in% ids) %>%
  arrange(mean_run_time)

race_means <- filtered_data %>%
  mutate(race_id = factor(race_id, levels = unique(race_id[order(mean_run_time)]))) %>%
  ggplot(aes(x = race_id, y = run_time, fill = race_id)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha = 0.6) +
  geom_jitter(color = "black", size = 0.4, alpha = 0.9) +
  theme_bw() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 11),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  xlab("race") +
  ylab("run time (s)")
# Make a boxplot of the run_times for each race_id

ggsave(race_means, filename = 'plots/race_means.pdf', width = 30, height = 20, units = 'cm')

race_z_scores <- filtered_data %>%
  mutate(race_id = factor(race_id, levels = unique(race_id[order(mean_run_time)]))) %>%
  ggplot(aes(x = race_id, y = z_score, fill = race_id)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha = 0.6) +
  geom_jitter(color = "black", size = 0.4, alpha = 0.9) +
  theme_bw() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 11),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  xlab("race") +
  ylab("run time (z-standardized per race)")
# Make a boxplot of the run_times for each race_id

ggsave(race_z_scores, filename = 'plots/race_z_scores.pdf', width = 30, height = 20, units = 'cm')

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


data_plot <- data_plot %>% mutate(gender_type = paste0(details_competition_type, ' - ', gender))

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
  data = data_plot,
  x = treatment,
  y = z_score,
  xlab = "Treatment",
  ylab = "Z-standardized (per race) run time",
  grouping.var = gender_type,
  type = "nonparametric",
  p.adjust.method = "bonferroni",
  # pairwise.display = "significant",
  plotgrid.args = list(nrow = 2),
)

ggsave(plot, filename = 'plots/z_score_boxplot.pdf', width = 30, height = 20, units = 'cm')
