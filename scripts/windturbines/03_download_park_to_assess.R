BASE_DIR<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
setwd(BASE_DIR)
source("scripts/windturbines/00_config.R")

COUNTRY<-"GLOBAL"
FILE_TURBINE_LOCATIONS=get_param(COUNTRY,
                                 "FILE_TURBINE_LOCATIONS")

wpd<-read_csv(FILE_TURBINE_LOCATIONS)

###select country###
COUNTRY<-"FR"
COUNTRY_ISO3<-"FRA"

windparks_country<-wpd %>% 
  filter(country==COUNTRY_ISO3 & fuel1 == "Wind") %>% 
  dplyr::select(name,latitude,longitude,commissioning_year) 

#path<-get_param(COUNTRY,"PATH_RAW_IMAGES_ASSESSMENT")



#unlink(list.dirs(path),
#      recursive = TRUE)

#windparks_country1<-windparks_country[5:nrow(windparks_country),]

for(i in 1:10){

    w<-windparks_country[i,]
    print(w$name)
    system.time(doSinglePark(w$name,
             w$latitude,
             w$longitude,
             RESOLUTION,
             COUNTRY))

}

