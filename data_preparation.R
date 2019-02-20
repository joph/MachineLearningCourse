#################REPOWER0.1 
#################This file has to be run only once to prepare the initial data
#################It may be rerun, if a finer grid is necessary



setwd("D:/google drive/diplomanden/Repowering")
source("functions.R")

###load data from IG-Windkraft website and save to disk
###is only necessary once or whenever the data needs to be updated
fileNameWindData<-"turbineData/turbines.feather"
importAndSaveIGWindData(fileNameWindData)

###read wind turbine data from disk
wind_turbines<-read_feather(fileNameWindData)

###create the convex hull of the points and save it to shape file
wind_turbines %>% group_by(Park) %>% dplyr::summarize(s=calculateHull(Long,Lat,Park))

###convert shape files to points
files<-paste("GIS/",sort(unique(wind_turbines$Park)),".shp",sep="")
resultingPoints<-sapply(files,convertShapeToPoints,0.003)

###bind results together
coordinates_possible_locations<-rbindlist(mapply(
  cbind.data.frame,
  resultingPoints,
  sort(unique(wind_turbines$Park)),
  SIMPLIFY=FALSE)) %>% as_tibble() 

names(coordinates_possible_locations)<-c("Long","Lat","Park")
coordinates_possible_locations<-coordinates_possible_locations %>% 
  mutate(Park_=as.character(Park),Type="additionalPoints",KW=NA,hubheight=NA,diameter=NA)

all_data<-bind_rows(coordinates_possible_locations,
                    wind_turbines %>% mutate(Type="originalPoints") %>% dplyr::select(Long,Lat,Park,Type,KW,hubheight=Nabenhoehe,diameter=Rotordurchmesser)) %>% 
  dplyr::select(Long,Lat,Park,Type,KW,hubheight,diameter) %>% arrange(Park) 



#######spatial join with a/k files
a<-raster("windatlas/a120_100m_Lambert.img")
k<-raster("windatlas/k120_100m_Lambert.img")


###spatial points of wind turbines, convert to projection of raster data
sp<-SpatialPoints(all_data[,c(1:2)])
projection(sp)<-"+proj=longlat +ellps=WGS84 +datum=WGS84"
sp<-spTransform(sp, projection(a))
points(sp)

###extract a and k parameters
a_extract<-extract(a, sp)
k_extract<-extract(k, sp)
all_data<-bind_cols(all_data,tibble(a_extract,k_extract))

####correct some missing data
all_data$a_extract[all_data$a_extract==0|all_data$k_extract==0]<-7
all_data$k_extract[all_data$a_extract==0|all_data$k_extract==0]<-2

all_data %>% filter(Park=="Albrechtsfeld") %>% ggplot(aes(x=Long,y=Lat)) + geom_point(aes(col=Type))


###create necessary GDX file
dir<-"additionalPoints"
all_data %>% dplyr::filter(Type==dir) %>% dplyr::group_by(Park) %>% 
  dplyr::summarize(s=writeOptmodelDistFiles(Long,Lat,max(Park),dir))

dir<-"originalPoints"
all_data %>% filter(Type==dir) %>% dplyr::group_by(Park) %>% 
  dplyr::summarize(s=writeOptmodelDistFiles(Long,Lat,max(Park),dir))

###save file to disk
write_feather(all_data,"turbineData/all_locations.feather")




