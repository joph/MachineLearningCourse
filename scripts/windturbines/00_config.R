BASE_DIR<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
setwd(BASE_DIR)
source("scripts/windturbines/functions.R")

###belgium should work...

SOURCE<-"Basemap"
url_source<-function(x,y,z){
  
  #GOOGLE  
  #return(paste0("http://mt0.google.com/vt/lyrs=s&hl=en&x=",x,"&y=",y,"&z=",z,""))

  #BASEMAP  
  return(paste0("https://maps2.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/",z,"/",y,"/",x,".jpeg"))

  
}
  

RESOLUTION<-19
COUNTRY_LIST<-c("CN","DE","FR","AT","BR","MIX","GLOBAL")

all_params<-list()

for(COUNTRY in COUNTRY_LIST){
  FILTER_WINDTURBINES_KW<-1000
  
  PATH_EXPERIMENT<-
    paste0(SOURCE,
           "/RESOLUTION",
           RESOLUTION,
           "/",
           COUNTRY,
           "/")
  
  PATH_TURBINE_LOCATIONS<-paste0(
    "data/turbineLocations",
    "/",
    COUNTRY,"/")
  
  FILE_TURBINE_LOCATIONS<-paste0(
    PATH_TURBINE_LOCATIONS,
    "windturbineLocations.csv")
  
  PATH_RAW_IMAGES_TURBINES<-paste0("data/aerialImages/",
                                   PATH_EXPERIMENT,
                                   "raw/Turbines/")
  
  PATH_RAW_IMAGES_NOTURBINES<-paste0("data/aerialImages/",
                                     PATH_EXPERIMENT,
                                     "raw/NoTurbines/")
  
  PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED<-paste0("data/aerialImages/",
                                   PATH_EXPERIMENT,
                                   "classified/Turbines/")
  
  PATH_RAW_IMAGES_NOTURBINES_MACHINE_CLASSIFIED<-paste0("data/aerialImages/",
                                     PATH_EXPERIMENT,
                                     "classified/NoTurbines/")
  
  PATH_RAW_IMAGES_ASSESSMENT<-paste0("data/aerialImages/",
                                   PATH_EXPERIMENT,
                                   "assessment/")
  
  #PATH_RAW_IMAGES_ASSESSMENT_TURBINES<-paste0("data/aerialImages/",
  #                                   PATH_EXPERIMENT,
  #                                   "assessment/turbines/")
  
  
  
  PATH_WINDPARK_LOCATIONS<-paste0(
    "data/windParks",
    "/",
    COUNTRY,"/")
  
  
  
  PATH_TEMP<-"data/temp/"
  
  
  PATH_LOCAL_TEMP<-"c:/temp/"
  
  
  PATH_QUALITYCHECK<-paste0(
    "qualityCheck/",
    PATH_EXPERIMENT)
  
  FILE_QUALITY_CHECK<-paste0(
    "qualityCheck/",
    PATH_EXPERIMENT,
    "qualityCheck.csv")
  
  PATH_ML_IMAGES_TURBINES_TRAIN<-paste0("data/aerialImages/",
                                        PATH_EXPERIMENT,
                                        "keras/train/Turbines/")
  
  PATH_ML_IMAGES_TURBINES_VALIDATION<-paste0("data/aerialImages/",
                                             PATH_EXPERIMENT,
                                             "keras/validation/Turbines/")
  
  
  PATH_ML_IMAGES_TURBINES_TEST<-paste0("data/aerialImages/",
                                       PATH_EXPERIMENT,
                                       "keras/test/Turbines/")
  
  
  PATH_ML_IMAGES_NOTURBINES_TRAIN<-paste0("data/aerialImages/",
                                          PATH_EXPERIMENT,
                                          "keras/train/NoTurbines/")
  
  PATH_ML_IMAGES_NOTURBINES_VALIDATION<-paste0("data/aerialImages/",
                                               PATH_EXPERIMENT,
                                               "keras/validation/NoTurbines/")
  
  
  PATH_ML_IMAGES_NOTURBINES_TEST<-paste0("data/aerialImages/",
                                         PATH_EXPERIMENT,
                                         "keras/test/NoTurbines/")
  
  
  
  
  dir.create("config",showWarnings = FALSE)
  
  
  
  params<-data.frame(SOURCE,
                     RESOLUTION,
                     COUNTRY,
                     PATH_EXPERIMENT,
                     FILTER_WINDTURBINES_KW)
  
  paths<-data.frame(
    PATH_TURBINE_LOCATIONS,
    PATH_RAW_IMAGES_TURBINES,
    PATH_RAW_IMAGES_NOTURBINES,
    PATH_WINDPARK_LOCATIONS,
    PATH_TEMP,
    PATH_LOCAL_TEMP,
    PATH_ML_IMAGES_TURBINES_TRAIN,
    PATH_ML_IMAGES_TURBINES_VALIDATION,
    PATH_ML_IMAGES_TURBINES_TEST,
    PATH_ML_IMAGES_NOTURBINES_TRAIN,
    PATH_ML_IMAGES_NOTURBINES_VALIDATION,
    PATH_ML_IMAGES_NOTURBINES_TEST,
    PATH_QUALITYCHECK,
    PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED,
    PATH_RAW_IMAGES_NOTURBINES_MACHINE_CLASSIFIED,
    PATH_RAW_IMAGES_ASSESSMENT
    #,
    #PATH_RAW_IMAGES_ASSESSMENT_TURBINES
  )
  
  sapply(unlist(paths),
         function(x){
           print(as.character(x))
           dir.create(as.character(x),showWarnings=FALSE,recursive = TRUE)})
  
  
  files<-data.frame(FILE_QUALITY_CHECK,
                    FILE_TURBINE_LOCATIONS)
  
  
  ####write config files
  params<-data.frame(params,
                     paths,
                     files)
  
  write_csv(params,paste0("config/params",COUNTRY,".csv"))
  all_params[[COUNTRY]]<-params
}

CURRENT_COUNTRY<-"BR"


