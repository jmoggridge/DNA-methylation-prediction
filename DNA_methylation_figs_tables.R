# DNA methylation figures and tables

library(tidyverse)
library(patchwork)

a <- read_rds("./figures/fig_islands.rds") +
  theme(legend.position = c(.8, .9))
b <- read_rds("./figures/fig_refgene_groups.rds") +
  theme(legend.position = 'none')
c <- read_rds("./figures/fig_chromosomes.rds")  +
  theme(legend.position = 'none')
panel_A <- ((c + b) / (a)) + plot_layout(guides = 'auto')
panel_B <- read_rds("./figures/fig_dinucleotide.rds") +
  theme(legend.position = 'na')

eda1 <- (panel_A | panel_B )
rm(a,b,c, panel_A, panel_B)

log_reg_cv <- read_csv("./results/logreg_cv_results.csv") %>% 
  transmute(
    mean_fit_time,
    C = round(param_logistic__C, 5),
    penalty = param_logistic__penalty,
    params,
    mean = mean_test_score,
    sd = std_test_score
  )
log_reg_cv_fig <- log_reg_cv %>% 
  filter(mean > 0.6) %>% 
  ggplot(aes(C, mean, color = penalty, fill = penalty)) +
  geom_path() +
  geom_point() +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd)) +
  scale_x_log10() +
  rcartocolor::scale_color_carto_d(palette = 1) +
  theme_bw() +
  theme(legend.position = c(0.7, 0.7)) +
  labs(y = 'AUC', x = 'C', color = NULL, shape = NULL, fill=NULL,
       subtitle = "Logistic regression ")

# linear SVM
svm4 <- read_csv("./results/svm4_cv_results.csv") %>% 
  transmute(C = param_svm__C,
            mean = mean_test_score,
            sd = std_test_score)

linear_svm_fig <- svm4 %>% 
  ggplot(aes(C, mean)) +
  geom_path(alpha = 0.95, color = 'gray') +
  geom_point() +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd)) +
  scale_x_log10() +
  labs(x = 'C', subtitle = "Linear SVM") +
  theme_bw()

# RBF SVM
svm_cv <- read_csv("./results/svm_cv_results.csv")
svm_cv2 <- read_csv("./results/svm2_cv_results.csv") 
svm_cv <- read_csv("./results/svm3_cv_results.csv") %>% 
  bind_rows(svm_cv2)%>% 
  transmute(
    mean_fit_time,
    C = round(param_svm__C,5),
    gamma = round(param_svm__gamma, 5),
    params,
    mean = mean_test_score,
    std_err = std_test_score
  ) 
svm_cv_fig <- svm_cv %>% 
  filter(mean>0.51) %>% 
  mutate(gamma = as_factor(gamma)) %>% 
  arrange(C) %>% 
  ggplot(aes(x = C, y = mean, color = gamma, fill = gamma)) +
  geom_point(alpha = 0.8, size = 0.75) +
  geom_path(alpha = 0.8) +
  scale_x_log10() +
  scale_color_viridis_d(option = 'C', begin = 0.1, end = 0.92) +
  scale_fill_viridis_d(option = 'C', begin = 0.1, end = 0.92) +
  theme_bw() +
  labs(x = 'C', subtitle = "RBF SVM")

# Random Forest 
rf_cv <- 
  bind_rows(
    read_csv("./results/rf2_cv_results.csv") %>% 
      mutate(data=2),
    read_csv("./results/rf_cv_results.csv") %>% 
      mutate(
        param_rf__min_samples_split = 2,
        param_rf__max_depth = 99999,
        data = 1
      )
  ) %>% 
  filter(param_rf__min_samples_leaf < 50) %>%
  transmute(
    estimators = factor(param_rf__n_estimators),
    mss = factor(param_rf__min_samples_split), 
    msl = factor(param_rf__min_samples_leaf),
    feat = factor(param_rf__max_features),
    max_depth = param_rf__max_depth,
    mean = mean_test_score,
    std_err = std_test_score,
  ) %>% 
  arrange(desc(mean))
best <- rf_cv[1,]

rf_cv_fig <- rf_cv %>% 
  ggplot(aes(y = factor(estimators), x = factor(msl), fill = mean)) +
  geom_tile() +
  geom_point(data=best, 
             aes(y = factor(estimators), x = factor(msl))) +
  facet_grid(feat ~ mss, labeller = label_both) +
  scale_fill_viridis_c(option = 'A') +
  labs(y = 'n trees', x = 'min. samples / leaf', fill = 'AUC') +
  theme_dark() +
  theme(strip.background = element_blank(),
        strip.text = element_text(color = 'black'))


best_logreg <- log_reg_cv %>% 
  filter(mean == max(mean)) %>% 
  transmute(name = "Logistic regression", mean, sd)
best_rbf_svm <- svm_cv %>% 
  filter(mean == max(mean)) %>% 
  transmute(name = "RBF SVM", mean, sd =std_err)
best_linear_svm <- svm4 %>% 
  filter(mean == max(mean)) %>% 
  transmute(name = "Linear SVM", mean, sd)
best_rf <- rf_cv %>% 
  filter(mean == max(mean)) %>% 
  transmute(name = "Random forest", mean, sd = std_err)


cv_results <- 
  bind_rows(best_logreg, best_linear_svm,best_rbf_svm, best_rf) %>% 
  transmute(name,
            CV_AUC = paste0(round(mean, 3),' (', round(sd, 3),')')
  )

validation_results <- 
  tribble(
    ~name,         ~AUC, ~Accuracy, ~Precision, ~Recall, ~F1,
    'Logistic regression', 0.985, 0.950, 0.909, 0.929, 0.919,
    'RBF SVM',             0.936, 0.945, 0.908, 0.912, 0.910,
    'Linear SVM',          0.932, 0.941, 0.895, 0.912, 0.903,
    'Random forest',       0.920, 0.938, 0.916, 0.875, 0.895,
  )
full_results <- 
  full_join(cv_results, validation_results, by = 'name') %>% 
  rename(Classifier = name)
table1 <- full_results %>% 
  kableExtra::kable(
    caption = "Performance metrics of best models in validation set prediction."
  )





## Paired t-test

logreg <- read_csv("./results/logreg_cv_results.csv") %>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains('split')) %>%
  pivot_longer(everything()) %>% 
  pull(value)
logreg 

svm3 <- read_csv("./results/svm_cv_results.csv")
svm2 <- read_csv("./results/svm2_cv_results.csv") 
svm <- read_csv("./results/svm3_cv_results.csv") %>% 
  bind_rows(svm_cv2, svm3)%>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains('split')) %>%
  pivot_longer(everything()) %>% 
  pull(value)
svm  

lin_svm <- read_csv("./results/svm4_cv_results.csv") %>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains('split')) %>%
  pivot_longer(everything()) %>% 
  pull(value)
lin_svm

rf <-  
  bind_rows(
    read_csv("./results/rf2_cv_results.csv") %>% 
      mutate(data=2),
    read_csv("./results/rf_cv_results.csv") %>% 
      mutate(data = 1)
  ) %>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains("test_score")) %>% 
  select(contains("split")) %>% 
  pivot_longer(everything()) %>% 
  pull(value)

rf

t.test(logreg, svm, paired = T)














