library(tidyverse)
library(ggplot2)
library(ggstatsplot)
library(ggpubr)
library(ggthemes)
library(afex)

data <- read_csv('data/data_z_score.csv')
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

data_plot <- data_differences %>%
             filter(!is.na(z_score_diff)) %>%
             pivot_longer(
               cols = c('treatment_true', 'treatment_false'),
               names_to = 'treatment',
               values_to = 'z_score') %>%
               mutate(treatment = ifelse(treatment == 'treatment_true', 'treated', 'not treated'))
 ggboxplot(data_plot,
              x = "treatment",
              y = "z_score",
              color = "treatment",
              palette = "jco",
              facet.by = "country",
              outlier.shape = NA,
              add = 'jitter',
    )
ggpaired(data_plot %>% filter(n_country > 350) %>% mutate(country = paste0(country, '(n = ', n_country, ')')),
         x = "treatment",
         y = "z_score",
         id="name",
         color = "treatment",
         palette = "jco",
         group = "country",
         facet.by = "country",
         outlier.shape = NA,
         line.color = "gray", line.size = 0.4,
         add = 'jitter',) +
        stat_compare_means(method = 'wilcox.test', paired = TRUE)


groupd <-  data_plot %>% filter(n_country > 350) %>% mutate(country = paste0(country, '(n = ', n_country, ')'))
groupd <-  data_plot %>% filter(rank_country < 11) %>% mutate(country = paste0(country, '(n = ', n_country, ')'))

grouped_ggbetweenstats(
  data=groupd,
  x = treatment,
  y = z_score,
  grouping.var = country,
  type= "nonparametric",
  p.adjust.method = "bonferroni",
  pairwise.display = "significant",
                       )

to_be_plotted <- data_plot %>% mutate(gender_type = paste0(details_competition_type, ' - ', gender ))

# Order gender_type factor type by alphabetical order
to_be_plotted$gender_type <-
  factor(to_be_plotted$gender_type,
         levels = c("Super G - Female",
                    "Slalom - Female",
                    "Giant Slalom - Female",
                    "Super G - Male",
                    "Slalom - Male",
                    "Giant Slalom - Male"
))

plot <- grouped_ggwithinstats(
  data=to_be_plotted,
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



# Get cols of data
colnames(data)
library(gt)
rx_adsl
custom_summary <- function(df, group_var, sum_var) {

  group_var <- rlang::ensym(group_var)
  sum_var <- rlang::ensym(sum_var)

  is_categorical <-
    is.character(eval(expr(`$`(df, !!sum_var)))) |
    is.factor(eval(expr(`$`(df, !!sum_var))))

  if (is_categorical) {

    category_lbl <-
      sprintf("%s, n (%%)", attr(eval(expr(`$`(df, !!sum_var))), "label"))

    df_out <-
      df |>
      dplyr::group_by(!!group_var)  |>
      dplyr::mutate(N = dplyr::n()) |>
      dplyr::ungroup() |>
      dplyr::group_by(!!group_var, !!sum_var) |>
      dplyr::summarize(
        val = dplyr::n(),
        pct = dplyr::n()/mean(N),
        .groups = "drop"
      ) |>
      tidyr::pivot_wider(
        id_cols = !!sum_var, names_from = !!group_var,
        values_from = c(val, pct)
      ) |>
      dplyr::rename(label = !!sum_var) |>
      dplyr::mutate(
        across(where(is.numeric), ~ifelse(is.na(.), 0, .)),
        category = category_lbl
      )

  } else {

    category_lbl <-
      sprintf(
        "%s (%s)",
        attr(eval(expr(`$`(df, !!sum_var))), "label"),
        attr(eval(expr(`$`(df, !!sum_var))), "units")
      )

    df_out <-
      df |>
      dplyr::group_by(!!group_var) |>
      dplyr::summarize(
        n = sum(!is.na(!!sum_var)),
        mean = mean(!!sum_var, na.rm = TRUE),
        sd = sd(!!sum_var, na.rm = TRUE),
        median = median(!!sum_var, na.rm = TRUE),
        min = min(!!sum_var, na.rm = TRUE),
        max = max(!!sum_var, na.rm = TRUE),
        min_max = NA,
        .groups = "drop"
      ) |>
      tidyr::pivot_longer(
        cols = c(n, mean, median, min_max),
        names_to = "label",
        values_to = "val"
      ) |>
      dplyr::mutate(
        sd = ifelse(label == "mean", sd, NA),
        max = ifelse(label == "min_max", max, NA),
        min = ifelse(label == "min_max", min, NA),
        label = dplyr::recode(
          label,
          "mean" = "Mean (SD)",
          "min_max" = "Min - Max",
          "median" = "Median"
        )
      ) |>
      tidyr::pivot_wider(
        id_cols = label,
        names_from = !!group_var,
        values_from = c(val, sd, min, max)
      ) |>
      dplyr::mutate(category = category_lbl)
  }

  return(df_out)
}

adsl_summary <-
  dplyr::filter(rx_adsl, ITTFL == "Y") |>
  (\(data) purrr::map_df(
    .x = dplyr::vars(AGE, AAGEGR1, SEX, ETHNIC, BLBMI),
    .f = \(x) custom_summary(df = data, group_var = TRTA, sum_var = !!x)
  ))()
rx_adsl_tbl <-
  adsl_summary |>
  gt(
    rowname_col = "label",
    groupname_col = "category"
  ) |>
  tab_header(
    title = "x.x: Demographic Characteristics",
    subtitle = "x.x.x: Demographic Characteristics - ITT Analysis Set"
  )

rx_adsl_tbl
adsl_summary %>% head(3) %>% dput()

competition_type_df <- data %>%
  mutate(group = details_competition_type) %>%
  group_by(group, treatment) %>%
  # Rename treatment to 'treated' and 'not treated'
  summarise(
            # median_wcp = median(total_wcp_nat_last),
            # fraction_treated = mean(treatment),
            median_run_time = median(run_time),
            mean_run_time = mean(run_time),
            sd_run_time = sd(run_time),
            n_run_time = n(),
            # median_z_score = median(z_score),
            # min_z_score = min(z_score),
            # max_z_score = max(z_score),
  )  %>% mutate(treatment = ifelse(treatment == 1, "Treated", "Not Treated")) %>%
  mutate(total_run_time = sum(n_run_time), fraction_run_time= n_run_time/total_run_time) %>% ungroup()

gender_df <- data %>%
   mutate(group = gender) %>%
  group_by(group, treatment) %>%
  # Rename treatment to 'treated' and 'not treated'
  summarise(
            # median_wcp = median(total_wcp_nat_last),
            # fraction_treated = mean(treatment),
            median_run_time = median(run_time),
            mean_run_time = mean(run_time),
            sd_run_time = sd(run_time),
            n_run_time = n(),
            # median_z_score = median(z_score),
            # min_z_score = min(z_score),
            # max_z_score = max(z_score),
  )  %>% mutate(treatment = ifelse(treatment == 1, "Treated", "Not Treated")) %>%
  mutate(total_run_time = sum(n_run_time), fraction_run_time= n_run_time/total_run_time) %>% ungroup()

# Discretize rank into three categories
rank_df <- data %>%
  mutate(rank_cat = cut(rank, breaks = c(0, 6, 11, 20, 60), labels = c("1 - 5", "5 - 10", "10 - 20", "> 30"))) %>%
  mutate(group = rank_cat) %>%
  group_by(group, treatment) %>%
  # Rename treatment to 'treated' and 'not treated'
  summarise(
            # median_wcp = median(total_wcp_nat_last),
            # fraction_treated = mean(treatment),
            median_run_time = median(run_time),
            mean_run_time = mean(run_time),
            sd_run_time = sd(run_time),
            n_run_time = n(),
            # median_z_score = median(z_score),
            # min_z_score = min(z_score),
            # max_z_score = max(z_score),
  )  %>% mutate(treatment = ifelse(treatment == 1, "Treated", "Not Treated")) %>%
  mutate(total_run_time = sum(n_run_time), fraction_run_time= n_run_time/total_run_time) %>% ungroup()

# Add rank_df to competition_type_df
combined_tibble <- bind_rows(competition_type_df, gender_df)


b <- combined_tibble %>% pivot_longer(cols = c(-group, -treatment), names_to = "label", values_to = "val") %>% separate(label, c("label", "category"), sep = "_")
bc <- b %>% ungroup() %>%  pivot_wider(names_from = c(treatment, category), values_from = val)

td <- b %>%  pivot_wider(names_from = c(treatment, label), values_from = val)
colnames(td)
# Pivot wider and split names from label column on '_'



td |> gt(
  rowname_col = "label",
  groupname_col = "group"
) |>
  tab_spanner(
    label = "Not treated",
    columns = c(`Not Treated_median`, `Not Treated_mean`, `Not Treated_sd`, `Not Treated_n`, `Not Treated_total`)
  ) |>
  tab_spanner(
    label = "Treated",
    columns = c("Treated_median", "Treated_mean", "Treated_sd", "Treated_n",
"Treated_total", "Treated_fraction")
  )

airquality_m <-
  airquality |>
  mutate(Year = 1973L) |>
  slice(1:10)