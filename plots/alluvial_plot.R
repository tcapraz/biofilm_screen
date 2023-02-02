library(ggplot2)
library(ggalluvial)
library(pheatmap)
library(dplyr)

pred = read.csv("/home/tuemay/biofilm/transfer_learning/structure_predictions.csv")

phylo_dist = read.csv("/home/tuemay/biofilm/phylo_data/biofilm_tree/dmat.csv", row.names = 1)
d = as.dist(phylo_dist)
plot(hclust(d), labels = FALSE)
pheatmap(phylo_dist)
hc = hclust(d)
clusters = cutree(hc, 10)

pred_sub = pred[pred$nt_num %in% names(clusters),]
pred_sub = pred_sub[pred_sub$day == "Day5",]
conditions = c("C22", "C21",  "C97")
pred_cond = pred_sub[pred_sub$cond %in% conditions,]
pred_cond = pred_cond[pred_cond$pred_biofilm!="discard",]
pred_cond$pred_biofilm[pred_cond$pred_biofilm=="non_structure"] <- "non"

pred_cond$phylo_class = NA
for (i in seq_along(clusters)){
  nt = names(clusters)[[i]]
  pred_cond$phylo_class[which(pred_cond$nt_num==nt)]= clusters[i]
}

nt = list()
for (i in seq_along(conditions)) {
  nt[[i]] = pred_cond$nt_num[pred_cond$cond == conditions[i]]
}
consensus_nt = Reduce(intersect, nt)
pred_cond =  pred_cond[pred_cond$nt_num %in% consensus_nt,]
pred_cond = pred_cond[!duplicated(pred_cond[,c("cond", "nt_num")]),]
table(pred_cond$pred_biofilm)
table(pred_cond$phylo_class)

pred_cond$phylo_class = as.factor(pred_cond$phylo_class)
pred_cond$pred_biofilm = as.factor(pred_cond$pred_biofilm)


mask = list()
for (i in seq_along(pred_cond$nt_num)){
  biofilms =pred_cond$pred_biofilm[which(pred_cond$nt_num==pred_cond$nt_num[[i]])]
  if (all(biofilms=="non")){
    mask[[i]] = FALSE
  }
  else{
    mask[[i]] = TRUE
  }
}

pred_cond = pred_cond[unlist(mask),]

cond_df = lapply(unique(pred_cond$cond), function(x){
  df =pred_cond[pred_cond$cond == x,]
  rownames(df) = df$nt_num
  df
})

cond_df[[2]] = cond_df[[2]][rownames(cond_df[[1]]),]
cond_df[[3]] = cond_df[[3]][rownames(cond_df[[1]]),]
#cond_df[[4]] = cond_df[[4]][rownames(cond_df[[1]]),]

allu_df = data.frame(control = cond_df[[2]]$pred_biofilm, EtOH = cond_df[[1]]$pred_biofilm, 
                     Hydroxyurea=cond_df[[3]]$pred_biofilm, phylo = cond_df[[1]]$phylo_class)
df =allu_df %>% group_by_all() %>% summarise(COUNT = n())
ggplot(df,   aes(axis1 = control, axis2 = EtOH, axis3=Hydroxyurea, 
                        y = COUNT))+
  geom_alluvium(aes(fill=phylo))+
  geom_stratum() +
  geom_text(stat = "stratum", aes(label = after_stat(stratum)), size=3) +
  theme_minimal() +
  theme(text = element_text(size=16), 
        panel.grid = element_blank())+
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))+
  ylab("Number of strains")+
  xlab("Condition")+
  scale_x_discrete(limits = c("control", "EtOH", "Hydroxyurea"), expand = c(.2, .05)) 
ggsave("condition_alluvial.png",dpi=400, width=10, height=10)
