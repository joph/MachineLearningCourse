library(tidyverse)
library(readxl)

setwd("/data/projects/windturbine-identification/MachineLearningCourse")
source("scripts/windturbines/00_config.R")


###download area data
download.file(url="http://api.worldbank.org/v2/en/indicator/AG.LND.TOTL.K2?downloadformat=excel",
              mode="wb",
              destfile="data/globalValidation/area.xls"
              )

filter_year <- "2017"

area1 <- read_excel(path="data/globalValidation/area.xls",sheet=1)

# select only relevant parts
area <- area1[4:nrow(area1),c(1,5:ncol(area1))]
names(area) <- area1[3,c(1,5:ncol(area1))]

area_filtered<-area %>% gather(year,area,-'Country Name') %>% mutate(area=as.numeric(area), country=`Country Name`) %>% 
  filter(year > "2000" & year < "2020") %>% group_by(country) %>% summarize(max_area=max(area,na.rm=TRUE))


area_filtered$country[area_filtered$country=="United States"]<-"USA"
area_filtered$country[area_filtered$country=="Egypt, Arab Rep."]<-"Egypt"
area_filtered$country[area_filtered$country=="Korea, Rep."]<-"Korea Rep"
area_filtered$country[area_filtered$country=="United Kingdom"]<-"UK"



# download query tool for wind and solar capacities
download.file("http://www.irena.org/IRENADocuments/IRENA_RE_electricity_statistics_-_Query_tool_v1.5.0.xlsm",
              destfile="data/globalValidation/irena.xlsm",
              mode="wb")

wcap1 <- read_excel(path="data/globalValidation/irena.xlsm",sheet=9)

# cut at column 37 because only before capacities (after production) and at row 6 to remove title
wcap1 <- wcap1[6:nrow(wcap1),1:37]

# remove every second column because empty
wcap1 <- wcap1[,c(1,2*(1:18))]
names(wcap1) <- c("country",2000:2017)

# remove empty lines
wcap <- wcap1[-which(is.na(wcap1$country)),]

wcap <- wcap %>% gather(year,cap,-country)


totcap_2017 <- wcap %>% filter(year == filter_year & country=="World") %>% summarize(capsum=sum(cap))

### we want to exclude around 1% of global capacity
filter_cap <- 329
#filter_cap <- 100
wcap %>% filter(year==filter_year & cap<filter_cap) %>% summarize(capsum_share=sum(cap)/totcap_2017)
wcap %>% filter(year==filter_year & cap<filter_cap) %>% nrow()

### remaining countries
regions<-c("World", "Africa","Asia","Europe","European Union","N America","S America","Middle East","Eurasia","Oceania","C America + Carib")

wcap_filtered<-wcap %>% filter(year==filter_year & cap>filter_cap) %>% filter(!(country %in% (regions))) %>% dplyr::select(country,cap)

###nmb_countries
wcap_filtered %>% nrow()




joined<-left_join(wcap_filtered,area_filtered)

joined %>% nrow()

joined %>% filter(is.na(max_area) == TRUE)

### Taiwan not found in world bank data...
joined <- joined %>% na.omit()

joined %>% summarize(area=sum(max_area,na.rm=TRUE))/10^6

### known turbines in brazil and in usa

#joined %>% filter(!(country %in% c("USA", "Brazil"))) %>% summarize(area=sum(max_area,na.rm=TRUE)/10^6)

### country abbreviations
download.file("https://datahub.io/core/country-list/r/data.csv",
              destfile="data/globalValidation/2digit_codes.csv")

abbreviations<-read_csv("data/globalValidation/2digit_codes.csv")

joined_abbreviations<-left_join(joined,abbreviations,by=c("country"="Name"))
joined_abbreviations$Code[joined_abbreviations$country=="Korea Rep"]<-"KP"
joined_abbreviations$Code[joined_abbreviations$country=="USA"]<-"US"
joined_abbreviations$Code[joined_abbreviations$country=="UK"]<-"GB"

joined_abbreviations$nmbTurbines<-0

for(i in 1:nrow(joined_abbreviations)){
  
  COUNTRY<-joined_abbreviations$Code[i]
  
  WIND_FILE<-get_param(COUNTRY,"FILE_OSM_TURBINE_LOCATIONS")
  
  turbines<-read_csv(WIND_FILE)
  n_turbines<-nrow(turbines)  
  
  joined_abbreviations$nmbTurbines[i]<-n_turbines

}

cap_nmbturbines_per_country<-joined_abbreviations

cap_nmbturbines_per_country<-cap_nmbturbines_per_country %>% mutate(avgCap=cap/nmbTurbines)

cap_nmbturbines_per_country %>% filter(avgCap>4) 
cap_nmbturbines_per_country %>% filter(avgCap>4) %>% summarize(sarea=sum(max_area))


cap_nmbturbines_per_country %>% ggplot(aes(x=country,y=avgCap)) + geom_col() + ylim(0,10)
