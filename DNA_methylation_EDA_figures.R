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
  theme(legend.position = c(0.7, 0.7)) +
  labs(subtitle = 'Relation to CpG island',
       x = '', y ='', color = '', fill = '')

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
  labs(y = 'proportion of sites', fill = '')
fig2
write_rds(fig2, "./figures/fig_chromosomes.rds")
rm(fig2)



meth <- read_csv("data/train.csv")
meth <- meth %>% 
  transmute(
    Id = Id,
    chromosome = CHR,
    UCSC_RefGene_Group,
    methylated = ifelse(Beta==1, 'methylated', 'unmethylated')
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
  labs(subtitle = 'RefGene tags', x = 'n CpG sites', y='', fill ='')
fig3  
write_rds(fig3, "./figures/fig_refgene_groups.rds")

rm(list = ls())


###
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
    cbind(
      Biostrings::letterFrequency(df$seq, letters = c('A', 'C','G'),
                                  as.prob = TRUE),
      Biostrings::dinucleotideFrequency(df$seq, as.prob = TRUE),
      Biostrings::trinucleotideFrequency(df$seq, as.prob = TRUE),
      Biostrings::oligonucleotideFrequency(df$seq, 4, as.prob = TRUE)
    ) %>%
    dplyr::select(-seq) 
  return(features.df)
}

meth_longseq_kmers <- meth %>% 
  generate_kmer_features() %>% 
  select(methylated, AA:TT)

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
fig4
write_rds(fig4, "./figures/fig_dinucleotide.rds")




meth_longseq_kmers %>% 
  select(methylated, CG, CA, GT, island) %>% 
  pivot_longer(CG:GT, names_to = 'dinucleotide',
               values_to = 'prop') %>% 
  ggplot(aes(x = prop*100, color = methylated, fill = methylated)) +
  geom_density(alpha = 0.1) +
  facet_grid(dinucleotide ~ island) +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  xlim(0, 15) +
  labs(x= '% composition', color ='',
       subtitle = "DNA composition over 2 kbp around CpG sites") +
  theme_minimal() +
  theme(panel.grid = element_blank(), 
        axis.text.y.left = element_blank(),
        axis.line.y = element_blank())


meth_longseq_kmers %>% 
  select(CA, CG, TG, methylated, island) %>% 
  mutate(CG_over_CA_TG = CG/(CA+TG)) %>% 
  ggplot(aes(CG_over_CA_TG, color = methylated, fill = methylated)) +
  geom_density(alpha = 0.1) +
  labs(x= "CG/(CA+TG)", color = '', 
       subtitle = "Dinucleotide composition 1kb flanking CpG sites") +
  theme_minimal() +
  facet_wrap(~island)
  

meth_shortseq_kmers <- 
  meth %>% 
  transmute(island,
            seq = Forward_Sequence,
            methylated) %>% 
  mutate(seq = str_remove_all(seq, '\\[|\\]')) %>% 
  generate_kmer_features() %>% 
  tibble() %>% 
  select(island, methylated, AA:TT) 


meth_shortseq_kmers %>% 
  pivot_longer(AA:TT, names_to = 'dinucleotide',
               values_to = 'prop') %>% 
  ggplot(aes(x = prop*100, 
             color = methylated, 
             fill = methylated)) +
  geom_density(alpha = 0.1, outline.type = 'full') +
  facet_wrap(~ dinucleotide, scales = 'free') +
  rcartocolor::scale_color_carto_d(palette = 2) +
  rcartocolor::scale_fill_carto_d(palette = 2) +
  labs(x = '% composition', 
       subtitle = "Dinucleotide composition 60 bp flanking CpG sites") +
  theme_minimal() +
  theme(panel.grid = element_blank(), 
        axis.text.y.left = element_blank(),
        axis.line.y = element_blank())





