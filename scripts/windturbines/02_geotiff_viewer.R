################## This is a small tool to manually rate the quality of the downloaded fotos

library(gtools)

CURRENT_COUNTRY<-"AT"

input_dir<-get_param(CURRENT_COUNTRY,
                     "PATH_RAW_IMAGES_TURBINES")

files<-list.files(input_dir) %>% mixedsort()

n<-length(files)

quality_check<-NULL

filename<-get_param(CURRENT_COUNTRY,
                    "FILE_QUALITY_CHECK")

if(file.exists(filename)){
 
   quality_check<-read_csv(filename)

   }else{
  
    windTurbines<-read_csv(get_param(CURRENT_COUNTRY,
                                     "FILE_TURBINE_LOCATIONS"))

    quality_check<-windTurbines %>% 
      filter(KW>get_param(CURRENT_COUNTRY,
                          "FILTER_WINDTURBINES_KW")) %>% 
      mutate(id=1:n(),quality=rep(100,n()))
}

for(i in 1:n){
  print(files[i])
  f<-files[i]
  f_nmb<-as.numeric(str_sub(f,1,-5))
  f1<-paste0(input_dir,f)
  r<-brick(f1)
  plotRGB(r)
  print(paste0("current rating is ",quality_check[f_nmb,]$quality))
  print("your rating of quality?")
  print("options: 1-100. -1 indicates that turbine is currently being built")
  print("q exits, n maintains current rating")
  q<-readline()
  if(q=="q"){
    break
  }
  if(q=="n"){
     next 
  }
  
  q<-as.numeric(q)  
  quality_check[f_nmb,]$quality<-q
  quality_check[f_nmb,]$id<-f_nmb
  
  
  
}


#####BEWARE BEFORE SAVING! IMPORTANT FILE MAY BE OVERWRITTEN!
write_csv(quality_check,filename)


#filename<-QUALITY_CHECK_FILE
#save<-"G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse/qualityCheck/Google/Resolution19/AT/qualityCheck_save.csv"
#qsave<-read_csv(save)

#fnames<-as.numeric(str_sub(files,1,-5))
#setdiff(1:1122,fnames)

#setdiff(1:1122,fnames)

#quality_check<-bind_rows(quality_check[1:329,] %>% mutate(quality=qsave$quality[1:329]),
#                         quality_check[330,],
#                         quality_check[331:549,] %>% mutate(quality=qsave$quality[330:548]),
#                         quality_check[550,],
#                         quality_check[551:553,] %>% mutate(quality=qsave$quality[549:551]),
#                         quality_check[554,],
#                         quality_check[555:617,] %>% mutate(quality=qsave$quality[552:614]),
#                         quality_check[618:626,],
#                         quality_check[627:1122,] %>% mutate(quality=qsave$quality[615:1110]))

