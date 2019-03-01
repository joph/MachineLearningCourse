BASE_DIR<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
setwd(BASE_DIR)
source("scripts/windturbines/functions.R")

SOURCE<-"Google"
RESOLUTION<-19
COUNTRY<-"BR"

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
          PATH_EXPERIMENT)

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
         PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED
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

write_csv(params,"config/params.csv")





