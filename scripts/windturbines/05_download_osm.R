setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../../")

source("scripts/windturbines/00_config.R")

require(ggplot2)
require(rgdal)
require(sf)

COUNTRIES<-c('IN', 'ES', 'GB', 'FR', 'BR', 'CA')

for(COUNTRY in COUNTRIES){

  PATH_OSM<-get_param(COUNTRY,"PATH_OSM")

  shape_file<-paste0(COUNTRY,".shp")

  shape <- readOGR(dsn = PATH_OSM) 

  windturbines<-st_as_sf(shape) %>% mutate(source=as.character(source)) 

  windturbines<-windturbines %>% filter(startsWith(source,"wind"))

  plot(windturbines)

  nrow(windturbines)

  coordinates<-st_coordinates(windturbines) %>% as_tibble()

  names(coordinates)<-c("Long", "Lat")

  write_csv(coordinates, get_param(COUNTRY,"FILE_OSM_TURBINE_LOCATIONS"))

  createWindTurbineImages(coordinates,
                        get_param(COUNTRY,
                                  "PATH_RAW_IMAGES_OSM"),
                        get_param(COUNTRY,
                                  "RESOLUTION"),
                        nmb=16)

  }