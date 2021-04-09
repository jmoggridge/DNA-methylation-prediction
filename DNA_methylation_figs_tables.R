# DNA methylation figures and tables

library(tidyverse)
library(patchwork)

meth <- read_csv("data/train.csv")
meth <- meth %>% 
  transmute(
    chromosome = CHR,
    island = Relation_to_UCSC_CpG_Island,
    feature = Regulatory_Feature_Group,
    methylated = ifelse(Beta==1, 'methylated', 'unmethylated')
  )
meth <- meth %>% 
  mutate(across(everything(),
                ~ ifelse(is.na(.x), 'None', .x)))

fig1 <- meth %>% 
  group_by(island, methylated) %>% 
  count() %>% 
  ggplot(aes(n, island, fill= methylated, color = methylated)) +
  geom_col() +
  rcartocolor::scale_color_carto_d(palette = 2) +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  theme_minimal() +
  theme(legend.position = c(.8, .9)) +
  labs(subtitle = 'Relation to CpG island',
       x = '', y ='', color = '', fill = '')

fig2 <- meth %>% 
  mutate(chromosome = as_factor(chromosome)) %>% 
  select(-island, -feature) %>% 
  group_by(chromosome) %>% 
  mutate(total_sites = n()) %>% 
  group_by(chromosome, methylated) %>% 
  mutate(proportion = n()/total_sites) %>% 
  distinct() %>% 
  ggplot(aes(chromosome, proportion, fill = methylated)) +
  geom_col() +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  labs(y = 'proportion of sites', fill = '') +
  theme_minimal() +
  theme(legend.position = 'none')

meth <- read_csv("data/train.csv")
fig3 <- meth %>% 
  transmute(
    Id = Id,
    chromosome = CHR,
    UCSC_RefGene_Group,
    methylated = ifelse(Beta==1, 'methylated', 'unmethylated')
  ) %>% 
  mutate(across(everything(), ~ ifelse(is.na(.x), 'None', .x))) %>% 
  mutate(terms = map(UCSC_RefGene_Group, ~str_split(.x, ';'))) %>% 
  unnest(terms) %>% 
  unnest(terms) %>% 
  group_by(Id, terms) %>% 
  distinct() %>% 
  count(methylated) %>% 
  ggplot(aes(y = terms, x = n, fill = methylated)) +
  geom_col() +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  theme_minimal() +
  theme(legend.position = 'null') +
  labs(subtitle = 'RefGene tags', x = 'n CpG sites', y='', fill ='')


panel_A <- ((fig2 + fig3) / (fig1)) + plot_layout(guides = 'auto')



# dinucleotide composition graph workflow

meth <- read_csv("./data/train.csv")
meth <- meth %>% transmute(
  Id, 
  Forward_Sequence,
  seq,
  methylated = ifelse(Beta==1, 'methylated', 'unmethylated'),
  island = Relation_to_UCSC_CpG_Island
)


# function generates sequence feature columns [(1-4)-mers] for classification
generate_kmer_features <- function(df){
  df <- as.data.frame(df)
  df$seq <- Biostrings::DNAStringSet(df$seq)
  features.df <- df %>%
    cbind(Biostrings::dinucleotideFrequency(df$seq, as.prob = TRUE)) %>%
    dplyr::select(-seq) 
  return(features.df)
}

# select dinucleotides only
meth_longseq_kmers <- meth %>% 
  generate_kmer_features() %>% 
  select(methylated, AA:TT)

# plot distributions of each dinucleotide
fig4 <- meth_longseq_kmers %>% 
  pivot_longer(AA:TT, names_to = 'dinucleotide',
               values_to = 'prop') %>% 
  ggplot(aes(x = prop*100, color = methylated, fill = methylated)) +
  geom_density(alpha = 0.1) +
  facet_wrap(~ dinucleotide) +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  rcartocolor::scale_color_carto_d(palette = 2) +
  labs(x= '% composition', color ='', fill = '', 
       subtitle = "DNA composition over 2 kbp around CpG sites") +
  theme_minimal() +
  xlim(0, 15) +
  theme(panel.grid = element_blank(), 
        axis.text.y.left = element_blank(),
        axis.line.y = element_blank())


eda1 <- (panel_A | fig4 )

rm(panel_A, fig1, fig2, fig3, fig4)

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
  arrange(desc(AUC)) %>% 
  kableExtra::kable(
    caption = "Performance metrics of best models in validation set prediction."
  )





## Paired t-tests

# first need to parse the split data from each cv results, get vector for each.

logreg <- read_csv("./results/logreg_cv_results.csv") %>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains('split')) %>%
  pivot_longer(everything()) %>% 
  pull(value)

svm3 <- read_csv("./results/svm_cv_results.csv")
svm2 <- read_csv("./results/svm2_cv_results.csv") 
svm <- read_csv("./results/svm3_cv_results.csv") %>% 
  bind_rows(svm_cv2, svm3)%>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains('split')) %>%
  pivot_longer(everything()) %>% 
  pull(value)

lin_svm <- read_csv("./results/svm4_cv_results.csv") %>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains('split')) %>%
  pivot_longer(everything()) %>% 
  pull(value)

rf <-  
  bind_rows(
    read_csv("./results/rf_cv_results.csv") %>%  mutate(data = 1),
    read_csv("./results/rf2_cv_results.csv") %>% mutate(data = 2)
  ) %>% 
  filter(mean_test_score == max(mean_test_score)) %>% 
  select(contains("test_score")) %>% 
  select(contains("split")) %>% 
  pivot_longer(everything()) %>% 
  pull(value)

paired_t_test <- function(x,y) t.test(unlist(x), unlist(y), paired=T)

# make df with columns of the 5 split scores for pairs of models
ttest_table <- 
  tribble(
    ~`Model 1`, ~scores1, ~`Model 2`, ~scores2,
    "Logistic regression", logreg, "RBF SVM", svm,
    "Logistic regression", logreg,  "Linear SVM", lin_svm,
    "Logistic regression", logreg, "Random forest", rf,
    "RBF SVM", svm, "Linear SVM", lin_svm,
    "RBF SVM", svm, "Random forest", rf,
    "Linear SVM", lin_svm, "Random forest", rf,
  ) %>% 
  # perform paired t-test
  mutate(t_test = map2(scores1, scores2, paired_t_test),
         `Mean difference` = map_dbl(t_test, "estimate"),
         `Mean difference` = round(`Mean difference`, 3),
         `p-value` = map_dbl(t_test, "p.value"),
         `p-value` = round(`p-value`, 3)
         ) %>% 
  select(-contains('scores'), -t_test)





















