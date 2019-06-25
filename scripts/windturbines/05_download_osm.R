BASE_DIR<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
setwd(BASE_DIR)
source("scripts/windturbines/00_config.R")

require(ggplot2)
require(rgdal)
require(sf)

COUNTRY<-"DE"

PATH_OSM<-get_param(COUNTRY,"PATH_OSM")

#zip_file<-paste0(PATH_OSM,"/",COUNTRY,".zip")

#unzip(zip_file, exdir=PATH_OSM)

shape_file<-paste0(COUNTRY,".shp")

shape <- readOGR(dsn = PATH_OSM) 

windturbines<-st_as_sf(shape) %>% mutate(source=as.character(source)) 

windturbines<-windturbines %>% filter(startsWith(source,"wind"))

plot(windturbines)

coordinates<-st_coordinates(windturbines) %>% as_tibble()

names(coordinates)<-c("Long", "Lat")

write_csv(coordinates, get_param(COUNTRY,"FILE_OSM_TURBINE_LOCATIONS"))

createWindTurbineImages(coordinates,
                        get_param(COUNTRY,
                                  "PATH_RAW_IMAGES_OSM"),
                        get_param(COUNTRY,
                                  "RESOLUTION"))
