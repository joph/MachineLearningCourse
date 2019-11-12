setwd("/data/projects/windturbine-identification/MachineLearningCourse")
source("scripts/windturbines/00_config.R")

library(maps)
library(spatstat)

predictions_osm<-feather("data/osm/all_predictions.feather") %>% as_tibble()


W <- owin(c(-180,180),c(-180,180))
pp1<-ppp(predictions_osm$lon,predictions_osm$lat,window=W)


n_<-360
qcount<-quadratcount(pp1,n_,n_)
int<-intensity(qcount) %>% as_tibble()

int<-int %>% mutate(x1=rep(-180:179,each=n_),y1=rep(179:-180,n_))

ggplot(int) +
  geom_point(aes(x=x1,y=y1,col=n),
             size=0.001) +
  scale_color_gradient(low="#FFFFFF",high=colorsERC3[1]) +

  
ggplot()+
    geom_polygon(data=world_map,
               aes(x=long,y=lat),
               fill=NA, 
               aes(group=group))
  



CURRENT_COUNTRY<-"US"

cs_locations<-read_csv(get_param(CURRENT_COUNTRY,
                                 "FILE_TURBINE_LOCATIONS"))

predictions_cs <- get_param(CURRENT_COUNTRY,
                            "PATH_RAW_IMAGES_TURBINES")

cs <- feather(paste0(predictions_cs,"all_predictions.feather")) %>% as_tibble()

tot<-bind_cols(cs_locations,cs)

write_csv(tot,"us_predictions.csv")

tot<-tot %>% mutate(class=ifelse(prediction==-1,3,
                            ifelse(prediction>(-0.1)&prediction<0.1,2,
                            ifelse(prediction>0.1,1,-1))))

tot %>% 
  group_by(class) %>% 
  dplyr::summarize(mcap=mean(t_cap,na.rm=TRUE),myear=mean(p_year,na.rm=TRUE),n=n())

tot %>% ggplot(aes(x=as.character(class),y=t_cap)) + geom_boxplot()
tot %>% ggplot(aes(x=as.character(class),y=p_year)) + geom_boxplot()

tot %>% filter(class==3) %>% ggplot(aes(x=xlong,y=ylat)) + geom_point(aes(col=as.character(class)))

tot %>% ggplot(aes(x=p_year)) + geom_histogram()

tot %>% filter(class==2) %>%  ggplot(aes(x=p_year)) + geom_histogram()


######validate turbines
COUNTRIES<-c("AT","DK","BR","US")

results<-NULL

for(CURRENT_COUNTRY in COUNTRIES){
  
  print(CURRENT_COUNTRY)

  osm_locations<-get_param(CURRENT_COUNTRY,"FILE_OSM_TURBINE_LOCATIONS")

  cs_locations<-read_csv(get_param(CURRENT_COUNTRY,
                                 "FILE_TURBINE_LOCATIONS"))

  predictions_osm<-feather("data/osm/all_predictions.feather") %>% as_tibble()

  osm<-predictions_osm %>% filter(country==CURRENT_COUNTRY) 

  predictions_cs <- get_param(CURRENT_COUNTRY,
                            "PATH_RAW_IMAGES_TURBINES")

  cs <- feather(paste0(predictions_cs,"all_predictions.feather"))

  cs<-cs %>% as_tibble() %>% mutate(Lon = Long, id=1:n()) %>% dplyr::select(Long,Lat,prediction)

  osm <- osm %>% mutate(Long = lon, Lat = lat,id=1:n()) %>% dplyr::select(Long,Lat,prediction)
  
  osm<- osm %>% na.omit()
  cs<- cs %>% na.omit()

  osm_cs<-match_closest_turbines(osm,cs,10,50,"osm","cs")

  osm_cs %>% group_by(has_twin,distance_twin,name) %>% dplyr::summarize(n=n())

  
  
  results<-bind_rows(results,
                     determine_quality(osm_cs,CURRENT_COUNTRY,"osm"),
                     determine_quality(osm_cs,CURRENT_COUNTRY,"cs"))
  
  
}

print(results)

colorsERC5<-c("#c62220", "#fbd7a8", "#7a6952", "#0d8085", "#f0c220")
colorsERC10<-c("#c72321","#861719","#fbd7a9","#ba9f7c","#7a6952","#6e9b9e","#0d8085","#19484c","#f0c320","#af8f19")

results %>% gather(var,val,-country,-name_select) %>% 
  filter(name_select=="cs") %>% 
  ggplot(aes(x=country,y=val)) + geom_bar(stat="identity",aes(fill=var),position="dodge") +
  scale_fill_manual(values=colorsERC10) +
  xlab("Country") + ylab("Number turbines")

