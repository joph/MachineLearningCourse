setwd("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
source("functions.R")
library(gtools)
library(tidyverse)

input_dir<-"data/testTB_Google/"



files<-list.files(input_dir) %>% mixedsort()

n<-length(files)
#quality_check<-tibble(id=1:n,quality=rep(0,n))



for(i in 253:n){
  print(files[i])
  f<-files[i]
  f1<-paste0(input_dir,f)
  r<-brick(f1)
  plotRGB(r)
  print("quality? (1-100)")
  q<-readline()
  if(q=="q"){
    break
  }else{
   q<-as.numeric(q)  
  }
  
  quality_check[i,]$quality<-q
  
  
  
}

write_feather(quality_check,"data/quality_check.feather")

