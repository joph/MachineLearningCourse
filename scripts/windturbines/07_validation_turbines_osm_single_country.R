setwd("/data/projects/windturbine-identification/MachineLearningCourse")
source("scripts/windturbines/00_config.R")



######validate turbines
CURRENT_COUNTRY<-"AT"

osm_locations<-get_param(CURRENT_COUNTRY,"FILE_OSM_TURBINE_LOCATIONS")

cs_locations<-read_csv(get_param(CURRENT_COUNTRY,
                                 "FILE_TURBINE_LOCATIONS"))

cs_locations<-cs_locations %>% 
  filter(KW>get_param(CURRENT_COUNTRY,
                      "FILTER_WINDTURBINES_KW"))

predictions_osm<-feather("data/osm/all_predictions.feather") %>% as_tibble()

predictions_filtered<-predictions_osm %>% filter(country==CURRENT_COUNTRY) 

predictions_cs <- get_param(COUNTRY,
                            "PATH_RAW_IMAGES_TURBINES")





######validate US turbines

#library(httr)
#set_config(config(ssl_verifypeer = 0L))
wind_turbines_us_osm<-read_csv(WIND_FILE)


download.file(url="https://eerscmap.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip",
              destfile="data/globalValidation/uswtdbCSV.zip",
              method="wget",
              mode="wb",
              extra=c("--no-check-certificate"))
unzip("data/globalValidation/uswtdbCSV.zip",
      exdir="data/globalValidation")
wind_turbines_us<-read_csv("data/globalValidation/uswtdb_v2_1_20190715.csv")

#prepare data
#osm data
osm <- data.frame(long=wind_turbines_us_osm$Long,
                lat=wind_turbines_us_osm$Lat)
coordinates(osm) <- ~long+lat
proj4string(osm) <- CRS("+init=epsg:4326")
osm_turbines<- spTransform(df1, CRS( "+init=epsg:4269" ) )

predictions<-feather("data/osm/all_predictions.feather") %>% as_tibble()

predictions$prediction %>% summary()

predictions %>% filter(prediction==-1) %>% nrow()

predictions_filtered<-predictions %>% filter(country=="US") 
nrow(osm)


#uswtdb
uswtdb<-data.frame(long=wind_turbines_us$xlong,
                   lat=wind_turbines_us$ylat)
coordinates(uswtdb)<- ~long+lat
proj4string(df2)<-CRS("+init=epsg:4269")
uswtdb_turbines<-df2

osm<-tibble(osm_turbines@coords[,1],
            osm_turbines@coords[,2])

uswtdb<-tibble(uswtdb_turbines@coords[,1],
               uswtdb_turbines@coords[,2])

names(osm)<-c("Lon","Lat")
names(uswtdb)<-c("Lon","Lat")



uswtdb<-bind_cols(uswtdb,wind_turbines_us)
osm<-bind_cols(osm,predictions_filtered)

uswtdb_osm<-match_closest_turbines(uswtdb,osm,10,30)
osm_uswtdb<-match_closest_turbines(osm,uswtdb,10,30)

uswtdb_osm_vars<-uswtdb_osm %>% filter(prediction>0) %>% 
  dplyr::select(lon_cs=lon_a,lat_cs=lat_a,lon_osm=lon_b,lat_osm=lat_b, dist,dist_eigen,dist_acc_meter,osm_p=prediction) %>% 
  mutate(has_partner=(dist<dist_eigen & dist_acc_meter<50),prediction_osm=osm_p>0.99)
  
uswtdb_osm_vars %>% filter(has_partner & prediction_osm) %>% nrow()
uswtdb_osm_vars %>% filter(has_partner & !prediction_osm) %>% nrow()
uswtdb_osm_vars %>% filter(!has_partner & prediction_osm) %>% nrow()
uswtdb_osm_vars %>% filter(!has_partner & !prediction_osm) %>% nrow()



uswtdb_osm %>% mutate(has_partner_osm=filt, exists_in_cs=1, exists_in_osm=prediction>0.99, )


uswtdb_osm %>% filter(prediction>-1) %>% filter(filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()
uswtdb_osm %>% filter(prediction>-1) %>% filter(!filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()

uswtdb_osm %>% filter(prediction>-1) %>% filter(Lon<0) %>%  ggplot(aes(x=Lon,y=Lat)) + geom_point(aes(col=prediction,size=p_year-1970))


osm_uswtdb %>%  filter(prediction>-1) %>% filter(filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()
osm_uswtdb %>%  filter(prediction>-1) %>% filter(!filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()

osm_uswtdb %>%  filter(prediction>-1) %>% filter(!filt) %>% nrow()
osm_uswtdb %>%  filter(prediction>-1) %>% filter(filt) %>% nrow()

uswtdb_osm %>%  filter(prediction>-1) %>% filter(!filt) %>% nrow()
uswtdb_osm %>%  filter(prediction>-1) %>% filter(filt) %>% nrow()


osm_uswtdb %>% filter(filt) %>% dplyr::select(prediction) %>% unlist() %>% hist()
osm_uswtdb %>% filter(!filt) %>% dplyr::select(prediction) %>% unlist() %>% hist()

uswtdb_osm %>% filter(prediction>-1) %>% ggplot(aes(x=filt,y=prediction)) + geom_boxplot()
osm_uswtdb %>% filter(prediction>-1) %>% ggplot(aes(x=filt,y=prediction)) + geom_boxplot()

graphix(uswtdb_osm)
graphix(osm_uswtdb)



######validate AT Turbines
WIND_FILE<-get_param("AT","FILE_OSM_TURBINE_LOCATIONS")

wind_turbines_at_osm<-read_csv(WIND_FILE)
nrow(wind_turbines_at_osm)

wind_at<-importAndSaveIGWindData("temp/tempturbines.csv") %>% as_tibble()
nrow(wind_at)

#prepare data
#osm data
osm <- data.frame(long=wind_turbines_at_osm$Long,
                  lat=wind_turbines_at_osm$Lat)
nrow(osm)
coordinates(osm) <- ~long+lat
proj4string(osm) <- CRS("+init=epsg:4326")

predictions_filtered<-predictions %>% filter(country=="AT") 


#uswtdb
igwind<-data.frame(long=wind_at$Long,
                   lat=wind_at$Lat)
coordinates(igwind)<- ~long+lat
proj4string(df2)<-CRS("+init=epsg:4326")

osm<-tibble(osm@coords[,1],
            osm@coords[,2])

igwind<-tibble(igwind@coords[,1],
               igwind@coords[,2])

names(osm)<-c("Lon","Lat")
names(igwind)<-c("Lon","Lat")


igwind<-bind_cols(igwind,wind_at)
osm<-bind_cols(osm,predictions_filtered)

igwind_osm<-match_closest_turbines(igwind,osm,10,30)
osm_igwind<-match_closest_turbines(osm,igwind,10,30)

igwind_osm %>% filter(prediction>-1) %>% filter(filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()
igwind_osm %>% filter(prediction>-1) %>% filter(!filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()

igwind_osm %>% filter(prediction>-1) %>% ggplot(aes(x=Lon,y=Lat)) + geom_point(aes(col=prediction ,size=as.numeric(Jahr)-2000))

osm_igwind %>% filter(prediction>-1) %>% filter(filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()
osm_igwind %>% filter(prediction>-1) %>% filter(!filt) %>% dplyr::select(prediction) %>% unlist() %>% summary()

igwind_osm %>% filter(prediction>-1) %>% ggplot(aes(x=filt,y=prediction)) + geom_boxplot()
osm_igwind %>% filter(prediction>-1) %>% ggplot(aes(x=filt,y=prediction)) + geom_boxplot()

graphix(igwind_osm)
graphix(osm_igwind)

