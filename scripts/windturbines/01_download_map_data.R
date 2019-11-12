setwd("/data/projects/windturbine-identification/MachineLearningCourse")
print(getwd())
source("scripts/windturbines/00_config.R")

checkIfTrue<-list()
checkIfTrue["US"]<-FALSE

RESOLUTIONS<-c(17:19)

COUNTRIES<-c("US")


  for(RESOLUTION in RESOLUTIONS){
  for(CURRENT_COUNTRY in COUNTRIES){
    for(i in 1:4)
    {
    print(CURRENT_COUNTRY)

    windTurbines<-read_csv(get_param(CURRENT_COUNTRY,
                                 "FILE_TURBINE_LOCATIONS",
                                 19))

    windTurbines_filtered<-windTurbines
    
    #%>% mutate(Park = 1:n())

#%>% 
#  filter(KW>get_param(CURRENT_COUNTRY,
#                      "FILTER_WINDTURBINES_KW"))

  print(nrow(windTurbines))
  print(nrow(windTurbines_filtered))
  print("windturbines download")
  createWindTurbineImages(windTurbines_filtered,
                        get_param(CURRENT_COUNTRY,
                                  "PATH_RAW_IMAGES_TURBINES",
                                  RESOLUTION),
                        get_param(CURRENT_COUNTRY,
                                  "RESOLUTION",
                                  RESOLUTION),
                        nmb=8)

  print("non windturbines download")
createNonWindTurbineImagesRandom(CURRENT_COUNTRY,
                                 windTurbines_filtered,
                                 nrow(windTurbines_filtered),
                                 get_param(CURRENT_COUNTRY,
                                           "PATH_RAW_IMAGES_NOTURBINES",
                                           RESOLUTION),
                                 get_param(CURRENT_COUNTRY,
                                           "PATH_WINDPARK_LOCATIONS",
                                           RESOLUTION),
                                 get_param(CURRENT_COUNTRY,
                                           "RESOLUTION",
                                           RESOLUTION),
                                 8,
                                 checkIfTrue[[CURRENT_COUNTRY]])
  }
  
}
}

