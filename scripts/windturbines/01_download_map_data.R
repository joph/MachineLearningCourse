
CURRENT_COUNTRY<-"AT"


windTurbines<-read_csv(get_param(CURRENT_COUNTRY,
                                 "FILE_TURBINE_LOCATIONS"))

windTurbines_filtered<-windTurbines %>% 
  filter(KW>get_param(CURRENT_COUNTRY,
                      "FILTER_WINDTURBINES_KW"))

nrow(windTurbines)
nrow(windTurbines_filtered)

createWindTurbineImages(windTurbines_filtered,
                        get_param(CURRENT_COUNTRY,
                                  "PATH_RAW_IMAGES_TURBINES"),
                        get_param(CURRENT_COUNTRY,
                                  "RESOLUTION"))

createNonWindTurbineImagesRandom(windTurbines_filtered,
                                 nrow(windTurbines_filtered),
                                 get_param(CURRENT_COUNTRY,
                                           "PATH_RAW_IMAGES_NOTURBINES"),
                                 get_param(CURRENT_COUNTRY,
                                           "PATH_WINDPARK_LOCATIONS"),
                                 get_param(CURRENT_COUNTRY,
                                           "RESOLUTION"))
