library(dplyr)
df <- readRDS("reproduction_papers.rds")

df_clean <- df |> select(-"referenced_works")

write.csv(df_clean, "reproduction_papers.csv", row.names=FALSE) 