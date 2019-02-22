
windTurbines<-read_csv(FILE_TURBINE_LOCATIONS)

windTurbines_filtered<-windTurbines %>% 
  filter(KW>FILTER_WINDTURBINES_KW)

nrow(windTurbines)
nrow(windTurbines_filtered)

createWindTurbineImages(windTurbines_filtered,
                        PATH_RAW_IMAGES_TURBINES,
                        RESOLUTION)

createNonWindTurbineImagesRandom(windTurbines_filtered,
                                 nrow(windTurbines_filtered),
                                 PATH_RAW_IMAGES_NOTURBINES,
                                 PATH_WINDPARK_LOCATIONS,
                                 RESOLUTION)
