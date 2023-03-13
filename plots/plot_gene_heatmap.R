library(ggplot2)
library(ggalluvial)
library(pheatmap)
library(dplyr)
library(stringr)
library(reshape2)
library(tidyverse)
library(ComplexHeatmap)
library(cultevo)
library(ape)
library(pheatmap)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# choose gene to classify strains on
gene="bcsQ"

pred = read.csv("../classification/structure_predictions.csv")
pred = pred[pred$day == "Day5",]

selected_cond = read.csv("../metadata/selected_conditions.csv", sep=";")
selected_cond = paste0("C",as.character(selected_cond$cond_num))

path_class = read.csv("../metadata/strains_used.csv", sep=" ")
nt_id = path_class$Strain.Identifier
path_class = path_class$Phenotype
names(path_class) = nt_id

pred = pred[pred$cond %in% selected_cond,]

phylo_dist = read.csv(paste0("../phylogeny/distances_per_gene/", gene, ".csv"), row.names = 1,  stringsAsFactors=FALSE)
phylo_dist[is.na(phylo_dist)] = 0
phylo_dist[phylo_dist ==""] = 0
phylo_dist[phylo_dist =="?"] = 1
phylo_dist <- sapply(phylo_dist, as.numeric )
rownames(phylo_dist) = colnames(phylo_dist)
# 
# phylo_dist[,"NT12106"]<-as.numeric(phylo_dist[,"NT12106"])
# 

phylo_dist = phylo_dist + t(phylo_dist)
p <- pheatmap::pheatmap(as.matrix(phylo_dist))


# png("/home/tuemay/biofilm/phylo_data/biofilm_tree/bcsE_tree.png",height=20, width = 25,units="in",res=1200)
# plot(hclust(d), labels = FALSE)
# 
# dev.off()

clusters = cutree(p$tree_col, k=2)

phylo_colors = c("brown4","lightskyblue2"  , "red", "gold", "blue",
                 "black", "mediumpurple4", "brown",  "grey",
                 "yellow4",  "deepskyblue4", "orange","sienna4")
names(phylo_colors)= unique(as.character(clusters))
c_anno = list(clusters = phylo_colors[1:length(unique(clusters))])
col_anno = data.frame(clusters = as.factor(clusters))
rownames(col_anno) = colnames(phylo_dist)
pheatmap::pheatmap(as.matrix(phylo_dist), annotation_col = col_anno, 
                   show_rownames = FALSE, show_colnames = FALSE, 
                   treeheight_row = 0, treeheight_col = 13, 
                   filename= paste0("phylodist_heatmap_",gene,".png"),
                   annotation_colors = c_anno)


cond_df = lapply(unique(pred$cond), function(x){
  df =pred[pred$cond == x,]
  df = df[!duplicated(df[,c("nt_num")]),]
  
  rownames(df) = df$nt_num
  df
})
# filter for interesting conditions
cond_df = lapply(cond_df, function(df){
  frac = sum(df$pred_biofilm == "non_structure")/dim(df)[1]
  if (frac > 0.9 & unique(df$cond) !="C21"){
    NULL
  } else{
    df
  }
})


cond_df[sapply(cond_df, is.null)] <- NULL
sel_cond = sapply(cond_df, function(x){
  unique(x$cond)
})
nt = lapply(cond_df, function(x){
  x$nt_num
})
consensus_nt = Reduce(intersect, nt)

map = seq_along(unique(pred$pred_biofilm))
names(map)=unique(pred$pred_biofilm)
pred_cond  = lapply(cond_df, function(x){
  cond = x[consensus_nt,"pred_biofilm"]
  # for (j in names(map)){
  #   cond <- ifelse(cond==j,  as.character(map[j]), cond)
  # }
  cond
})
mat = as.data.frame(t(data.frame(do.call(rbind, pred_cond))))

rownames(mat) = consensus_nt
colnames(mat) = sel_cond

path_class = path_class[rownames(mat)]
#mat = mat [rownames(mat)%in% names(clusters),]

mat =mat[rowSums(mat != "non_structure")!=0,]
c("1"= "brown4","2"="lightskyblue2"  , "3" ="red", "4"="green","5"= "blue",
  "6"= "black","7" = "mediumpurple4", "8" = "brown", "10" = "grey",
  "11" = "yellow4", "12"= "deepskyblue4", "13" = "orange")

ha = rowAnnotation(phylogenetic_class = as.factor(clusters[rownames(mat)]),
                   col = list(phylogenetic_class=phylo_colors[1:length(unique(clusters))]))
#ha = rowAnnotation(phylogenetic_class = as.factor(path_class[rownames(mat)]), 
#                   col = list(phylogenetic_class=c("Commensal strain" ="red", "Laboratory strain"="green","Pathogenic strain"= "blue")))
kelly.colours <- c( "gray13","white", "gold2", "plum4", "darkorange1", "yellow4", 
                    "firebrick", "burlywood3", "gray51", "springgreen4","lightskyblue2", 
                    "deepskyblue4",  "mediumpurple4", "orange", "maroon", 
                    "yellow3", "brown4", "yellow4", "sienna4", "chocolate", "gray19")
dev.off()
png(paste0("/home/tuemay/biofilm/phylo_data/biofilm_tree/heatmap_", gene,".png"),height=15, width = 7,units="in",res=1200)

Heatmap(as.matrix(mat), clustering_distance_rows = hammingdists, 
        clustering_distance_columns = hammingdists, col = kelly.colours[1:length(unique(unlist(c(mat))))], 
        right_annotation = ha,heatmap_height = unit(0.5, "cm")*nrow(mat),
        heatmap_width = unit(0.5, "cm")*ncol(mat), border=TRUE,
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.rect(x = x, y = y, width = width, height = height, 
                    gp = gpar(col = "grey", fill = NA))})
dev.off()
