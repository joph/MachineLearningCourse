setwd("D:/google drive/diplomanden/Repowering")
source("functions.R")

wind_turbines<-read_feather("turbineData/all_locations.feather")
all_data<-read_feather("turbineData/all_locations.feather")

###count locations in two scenarios
all_data %>% group_by(Type) %>% dplyr::summarize(n())

###double existing points to calculate baseline power generation
all_data<-bind_rows(all_data,
          all_data %>% filter(Type=="originalPoints") %>% mutate(Type="baseline"))

###simulate production at all points
###in particular of new turbines

###Here is decided, which turbine size in kw, which hubheight and which diameter of the wind-turbine
###should be simulated
all_data_prepare_run<-simAtAllPoints(all_data,4200,130,135)

write_excel_csv(all_data_prepare_run,"turbineData/all_locations.csv")

all_data_prepare_run %>% dplyr::group_by(Type) %>% summarize(sumProd=sum(annualProduction,na.rm=TRUE)/10^6)

###scenario may be "additionalPoints" or "originalPoints"
###the number in runScenario indicates the minimum distance 
###between turbines in km
scenario<-"additionalPoints"
p_out_1<-runScenario(scenario,0.6,all_data_prepare_run %>% filter(Type==scenario),FALSE)

scenario<-"originalPoints"
p_out_2<-runScenario(scenario,0.6,all_data_prepare_run %>% filter(Type==scenario),FALSE)

final<-bind_rows(p_out_1 %>% mutate(Scenario="Additional"),
                 p_out_2 %>% mutate(Scenario="Original"),
                 all_data_prepare_run %>% filter(Type=="baseline") %>% mutate(Scenario="Baseline"))

#repowering result
final %>% dplyr::group_by(Scenario,Type) %>% 
  dplyr::summarize(sumProd=sum(annualProduction,na.rm=TRUE)/10^6,sumLocs=n())

final %>% dplyr::filter(Type=="Opt" & (Scenario=="Additional"|Scenario=="Original")) %>% write_excel_csv("scenarios.csv")


#for debugging purposes
tt<-final %>% dplyr::group_by(Scenario,Type,Park) %>% dplyr::summarize(sum=n()) %>% spread(Type,sum) %>% 
  dplyr::select(Scenario,Park,Opt) %>% spread(Scenario,Opt)


