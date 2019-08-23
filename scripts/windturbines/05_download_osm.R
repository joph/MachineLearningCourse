#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../../")

source("scripts/windturbines/00_config.R")

COUNTRIES<-c("EG", "MA", "ZA", "CN", "IN", "JP", "KP", "PK", "PH", "TH", "CR", "TR", 
                "AT", "BE", "BG", "HR", "DK", "FI", "FR", "DE", "GR", "IE", "IT", "LT", 
                "NL", "NO", "PL", "PT", "RO", "ES", "SE", "GB", "UA", "CA", "MX", "US", "AU", "NZ", "BR", "CL", "UY")

#COUNTRIES<-c("US", "AU", "NZ", "BR", "CL", "UY")

tot_turbines<-0
tot_downloaded<-0

for(COUNTRY in COUNTRIES){


  print(COUNTRY)  
  
  PATH_OSM<-get_param(COUNTRY,"PATH_OSM")

  shape_file<-paste0(COUNTRY,".shp")

  shape <- readOGR(dsn = PATH_OSM) 

  windturbines<-st_as_sf(shape) %>% mutate(source=as.character(source)) 

  windturbines<-windturbines %>% filter(startsWith(source,"wind"))
  

  #plot(windturbines)

  #nrow(windturbines)

  coordinates<-st_coordinates(windturbines) %>% as_tibble()

  names(coordinates)<-c("Long", "Lat")

  write_csv(coordinates, get_param(COUNTRY,"FILE_OSM_TURBINE_LOCATIONS"))

  dir_osm<-get_param(COUNTRY,
                     "PATH_RAW_IMAGES_OSM")
  
    
  if(length(list.files(dir_osm))< nrow(windturbines)){

    createWindTurbineImages(coordinates,
                      get_param(COUNTRY,
                                  "PATH_RAW_IMAGES_OSM"),
                        get_param(COUNTRY,
                                  "RESOLUTION"),
                        nmb=24)
  }
  

  print("nmb turbines in dataset:")
  tot_turbines<-nrow(windturbines)+tot_turbines
  print(tot_turbines)
  
    
  print("tiles downloaded:")  
  tot_downloaded<-list.files(dir_osm) %>% length() + tot_downloaded
  print(tot_downloaded)
  
  print("diff:")
  print(tot_turbines-tot_downloaded)
  

}
