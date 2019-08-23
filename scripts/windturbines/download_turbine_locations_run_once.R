#### RUN ONCE TO DOWNLOAD AND UPDATE DATA FOR BRAZIL/AUSTRIA
base_dir<-("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
setwd(base_dir)
source("scripts/windturbines/functions.R")


############# AUSTRIA
COUNTRY="AT"

FILE_TURBINE_LOCATIONS=get_param(COUNTRY,
                                 "FILE_TURBINE_LOCATIONS")

windTurbines<-importAndSaveIGWindData(FILE_TURBINE_LOCATIONS) %>% 
  as_tibble()

###### BRAZIL Wind turbine locations - update table to incorporate KW Column
COUNTRY="BR"

FILE_TURBINE_LOCATIONS=get_param(COUNTRY,
                                 "FILE_TURBINE_LOCATIONS")

#download.file("http://datasets.wri.org/dataset/540dcf46-f287-47ac-985d-269b04bea4c6/resource/27c271ef-63c3-49c5-a06a-f21bb7b96371/download/globalpowerplantdatabasev110",
#              destfile=FILE_TURBINE_LOCATIONS)

# manually pre-processed...

windTurbines<-read_csv(paste0(FILE_TURBINE_LOCATIONS,".all")) %>% 
  mutate(KW=POT_MW*1000,Park=CEG)

write_csv(windTurbines,FILE_TURBINE_LOCATIONS) 

###############GLOBAL POWER PLANT DATABASE
COUNTRY<-"GLOBAL"

FILE_TURBINE_LOCATIONS=get_param(COUNTRY,
                                 "FILE_TURBINE_LOCATIONS")

f<-paste0(FILE_TURBINE_LOCATIONS,".zip")

download.file("http://datasets.wri.org/dataset/540dcf46-f287-47ac-985d-269b04bea4c6/resource/27c271ef-63c3-49c5-a06a-f21bb7b96371/download/globalpowerplantdatabasev110",
              destfile=f,
              mode="wb"
              )

unzip(f)

file.copy("global_power_plant_database.csv",
          FILE_TURBINE_LOCATIONS)

file.remove("global_power_plant_database.csv")


###############GLOBAL POWER PLANT DATABASE
COUNTRY<-"US"

FILE_TURBINE_LOCATIONS=get_param(COUNTRY,
                                 "FILE_TURBINE_LOCATIONS")




f<-paste0(FILE_TURBINE_LOCATIONS,".zip")

download.file(url="https://eerscmap.usgs.gov/uswtdb/assets/data/uswtdbCSV.zip",
              destfile=f,
              method="wget",
              mode="wb",
              extra=c("--no-check-certificate"))
unzip(f)

file.copy("uswtdb_v2_1_20190715.csv",
          FILE_TURBINE_LOCATIONS)

file.remove("uswtdb_v2_1_20190715.csv")

tab<-read_csv(FILE_TURBINE_LOCATIONS)


tab<-tab %>% mutate(KW=t_cap, Long=xlong, Lat=ylat) 
write_csv(tab,FILE_TURBINE_LOCATIONS)



