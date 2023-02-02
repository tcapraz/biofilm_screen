library(ggplot2)
library(ggalluvial)
library(pheatmap)
library(dplyr)
library(stringr)
library(reshape2)
library(tidyverse)
pred = read.csv("/home/tuemay/biofilm/transfer_learning/structure_predictions.csv")
pred = pred[pred$day == "Day5",]

cond_df = lapply(unique(pred$cond), function(x){
  df =pred[pred$cond == x,]
  df = df[!duplicated(df[,c("nt_num")]),]
  
  rownames(df) = df$nt_num
  df
})

nt = lapply(cond_df, function(x){
  x$nt_num
})
consensus_nt = Reduce(intersect, nt[1:30])
cond_df = cond_df[1:30]
map = seq_along(unique(pred$pred_biofilm))
names(map)=unique(pred$pred_biofilm)
pred_cond  = lapply(cond_df, function(x){
  cond = x[consensus_nt,"pred_biofilm"]
  for (j in names(map)){
    cond <- ifelse(cond==j,  as.character(map[j]), cond)
  }
  cond
})
mat = as.data.frame(t(data.frame(do.call(rbind, pred_cond))))

rownames(mat) = consensus_nt
colnames(mat) = unique(pred$cond)[1:30]
ph = pheatmap(data.matrix(mat), fontsize = 6, width=10, height=10, filename="heatmap.png", dpi=400)
row_order = ph$tree_row$order
col_order = ph$tree_col$order
mat = mat[row_order, col_order]
mat$strain = rownames(mat)
dat3 = pivot_longer(mat, cols = colnames(mat)[1:30])
dat3$strain = factor(dat3$strain, levels=rownames(mat))
dat3$name = factor(dat3$name, levels=colnames(mat)[1:30])
for (j in names(map)){
  dat3$value <- ifelse(dat3$value==as.character(map[j]),j, dat3$value)
}
kelly.colours <- c( "gray13","gray95", "gold2", "plum4", "darkorange1", "lightskyblue2", 
                   "firebrick", "burlywood3", "gray51", "springgreen4", "lightpink2", 
                   "deepskyblue4", "lightsalmon2", "mediumpurple4", "orange", "maroon", 
                   "yellow3", "brown4", "yellow4", "sienna4", "chocolate", "gray19")
ggplot(dat3, aes(name, strain)) + geom_tile(aes(fill = value),
                                                colour = "grey") + scale_fill_manual(values=kelly.colours[1:13])+
  theme_minimal() +
  theme(text = element_text(size=8))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
      axis.line = element_line(colour = "black"))+
  ylab("Strains")+
  xlab("Condition")
ggsave("heatmap.png", height=15, width = 7, dpi=400)
