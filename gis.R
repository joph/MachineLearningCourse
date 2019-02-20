setwd("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse")
source("functions.R")


###### AUSTRIA

windTurbines<-importAndSaveIGWindData("data/windturbines.feather") %>% 
  as_tibble()

###remove long/lat duplicates
longlat<-windTurbines %>% mutate(n=1:n()) %>% dplyr::select(Long,Lat,n)
windTurbines_filtered<-windTurbines %>% 
  filter(!(row_number() %in% longlat[duplicated(longlat[,1:2]),]$n)) %>% 
  filter(KW>1000)


#%>% filter(!(row_number() %in% wrong_turbines))


###for legacy reasons
#windTurbines_filtered1<-windTurbines %>% filter(KW>1000&Jahr!=""&Long<=16) %>% 
#  mutate(Jahr=as.numeric(Jahr)) %>% filter(Jahr<2016) 

#windTurbines_filtered<-bind_rows(windTurbines_filtered,windTurbines_filtered1)

#no_turbines<-c(5,29,54,62,75,80,83,84,130,132,134,135,141,152,156,158,160,162,176,195,221,22,223,234,244,245,256,258,277,280,292,297,316,329,330,387,393,430,434,435,437,442,446,453,466,470,483,484,485,487,488,491,494,499,511,515,516,517,518,520,521,522,527,535,553,554,55,556,557,558,559,560,561,562,567,577,595,598,614,615,616,617,618,619,620,621,622,623,625,626,636,640,652,653,654,655,670,718,723,736,773,785,810,811,812,813,814,815,816,817,818,819,820,821,833,834)
#little_turbines<-c(42,50,53,55,56,57,65,7499,102,76,81,86,88,123,126,138,140,145,147,148,153,161,165,166,167,172,175,183,184,185,188,214,218,230,233,238,242,254,267,271,291,299,306,328,342,349,365,373,398,400,402,407,423,433,436,445,458,462,467,472,473,474,478,480,486,492,496,512,514,519,528,529,531,532,533,538,543,544,569,586,590,601,634,635,645,649,668,669,679,682,692,703,705,713,719,720,726,748,749,757,758,763,768,770,771,786,788,797,828)

#both<-c(no_turbines,little_turbines)

#windTurbines_filtered<-windTurbines_filtered %>% 
#  filter(!(row_number() %in% both)) 

nrow(windTurbines)
nrow(windTurbines_filtered)

out_windturbines<-"data/testTB_Google/"

unlink(paste0(out_windturbines,"*"))
createWindTurbineImages(windTurbines_filtered,
                        out_windturbines,19)

out_no_windturbines<-"data/testNoTB_Google/"
unlink(paste0(out_no_windturbines,"*"))
createNonWindTurbineImagesRandom(windTurbines_filtered,
                                 nrow(windTurbines_filtered),
                                 out_no_windturbines,"data/parks/",19)



###### BRAZIL

brazilturbines<-read_csv('data/brazilTurbineShape/brazilturbines_2.csv')
brazilturbines<-brazilturbines %>% filter(POT_MW>1 & Operacao == "SIM") 

createWindTurbineImages(brazilturbines,
                        "data/testTB_BR_Google/",18)

createNonWindTurbineImagesRandom(brazilturbines,
                                 nrow(windTurbines_filtered),
                                 "data/testNoTB_BR_Google/","data/parks/brazil/",18)


