library(tidyverse)
library(ggplot2)
library(zoo)


setwd("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse/presentations/scripts")

erc_colors<-c("#C72321","#6E9B9E")

val_acc<-read_csv2("../data/validation_loss_accuracy_unfreezed_model.csv")

for(i in 30:40){
  val_acc[c(i,40),]<-val_acc[29,]
}

val_acc$Epoch[30:40]<-30:40


val_acc_gathered<-val_acc %>% gather(Scen,Value,-Epoch) %>% 
  mutate(Type=ifelse(str_sub(Scen,1,1)=="L","Loss","Accuracy")) %>% 
  mutate(Value=as.numeric(Value)) %>% group_by(Scen,Type) %>% 
  mutate(rollmean=rollmean(Value,5,na.pad=TRUE,align="left")) %>% 
  filter(Epoch<30) %>% ungroup()


val_acc_gathered %>% ggplot(aes(x=Epoch,Value))+geom_line(aes(col=Scen),size=1,linetype=2) +
  scale_color_manual(values=c(erc_colors,erc_colors)) + 
  geom_line(aes(y=rollmean,col=Scen),size=2)+
  facet_wrap(Type~.,scales = "free")+
  theme_bw()

ggsave("../figures/accuracy_loss.png")
