#### RUN ONCE TO DOWNLOAD AND UPDATE DATA FOR BRAZIL/AUSTRIA
base_dir<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
source("scripts/windturbines/functions.R")


############# AUSTRIA
COUNTRY="AT"



PATH_TURBINE_LOCATIONS=paste0(
  "data/turbineLocations",
  "/",
  COUNTRY,"/")

FILE_TURBINE_LOCATIONS=paste0(
  TURBINE_LOCATIONS_PATH,
  "windturbineLocations.csv")

dir.create(PATH_TURBINE_LOCATIONS,
           recursive=TRUE,
           showWarnings = FALSE)


###### Country specific imports
###### AUSTRIA Wind turbine locations
windTurbines<-importAndSaveIGWindData(TURBINE_LOCATIONS_FILE) %>% 
  as_tibble()

###### BRAZIL Wind turbine locations - update table to incorporate KW Column
COUNTRY="BR"

PATH_TURBINE_LOCATIONS=paste0(
  "data/turbineLocations",
  "/",
  COUNTRY,"/")

FILE_TURBINE_LOCATIONS=paste0(
  PATH_TURBINE_LOCATIONS,
  "windturbineLocations.csv")

dir.create(PATH_TURBINE_LOCATIONS,
           recursive=TRUE,
           showWarnings = FALSE)

windTurbines<-read_csv(paste0(FILE_TURBINE_LOCATIONS,".all")) %>% 
  mutate(KW=POT_MW*1000,Park=CEG)

write_csv(windTurbines,TURBINE_LOCATIONS_FILE)  
