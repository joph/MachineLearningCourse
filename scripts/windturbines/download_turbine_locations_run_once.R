#### RUN ONCE TO DOWNLOAD AND UPDATE DATA FOR BRAZIL/AUSTRIA
base_dir<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
source("scripts/windturbines/functions.R")


############# AUSTRIA
COUNTRY="AT"



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
windTurbines<-importAndSaveIGWindData(TURBINE_LOCATIONS_FILE) %>% 
  as_tibble()

###### BRAZIL Wind turbine locations - update table to incorporate KW Column
COUNTRY="BR"

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

windTurbines<-read_csv(TURBINE_LOCATIONS_FILE) %>% 
  mutate(KW=POT_MW*1000,Park=CEG)
write_csv(windTurbines,TURBINE_LOCATIONS_FILE)  
