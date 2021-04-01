library(tidyverse)
library(patchwork)

meth <- read_csv("data/train.csv")

meth <- meth %>% 
  transmute(
    chromosome = CHR,
    island = Relation_to_UCSC_CpG_Island,
    feature = Regulatory_Feature_Group,
    methylated = ifelse(Beta==1, TRUE, FALSE)
  )
meth <- meth %>% 
  mutate(across(everything(), ~ ifelse(is.na(.x), 'None', .x)))

fig1 <- meth %>% 
  group_by(island, methylated) %>% 
  count() %>% 
  ggplot(aes(n, island, fill= methylated, color = methylated)) +
  geom_col() +
  rcartocolor::scale_color_carto_d(palette = 2) +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  theme_light() +
  theme(legend.position = 'NA') +
  labs(subtitle = 'Relation to CpG island', x = '', y ='')

fig1
write_rds(fig1, "./figures/fig_islands.rds")
rm(fig1)

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
  theme_minimal() +
  labs(subtitle = 'Chromosome', y = 'proportion of sites')
fig2
write_rds(fig2, "./figures/fig_chromosomes.rds")
rm(fig2)



meth <- read_csv("data/train.csv")

meth <- meth %>% 
  transmute(
    Id = Id,
    chromosome = CHR,
    UCSC_RefGene_Group,
    methylated = ifelse(Beta==1, TRUE, FALSE)
  ) %>% 
  mutate(across(everything(), ~ ifelse(is.na(.x), 'None', .x)))

fig3 <- meth %>% 
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
  labs(subtitle = 'RefGene tags', x = 'n CpG sites')
fig3  
write_rds(fig3, "./results/fig_refgene_groups.rds")

rm(list = ls())

meth <- read_csv("./data/train.csv")
meth %>% transmute(
  Id, 
  seq,
  Beta
)

library















































