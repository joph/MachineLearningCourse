base_dir<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
source("scripts/windturbines/functions.R")

SOURCE="Google"
RESOLUTION="Resolution19"
COUNTRY="At"

FILTER_WINDTURBINES_KW=1000

PATH_EXPERIMENT=
  paste0(SOURCE,
         "/",
         RESOLUTION,
         "/",
         COUNTRY,
         "/")

TURBINE_LOCATIONS_PATH=paste0(
  "data/turbineLocations",
  "/",
  COUNTRY,"/")

TURBINE_LOCATIONS_FILE=paste0(
  TURBINE_LOCATIONS_PATH,
  "windturbineLocations.csv")

dir.create(TURBINE_LOCATIONS_PATH,
           recursive=TRUE,
           showWarnings = FALSE)


###### Country specific imports
###### AUSTRIA Wind turbine locations
#windTurbines<-importAndSaveIGWindData(TURBINE_LOCATIONS_FILE) %>% 
#  as_tibble()

###### BRAZIL Wind turbine locations




