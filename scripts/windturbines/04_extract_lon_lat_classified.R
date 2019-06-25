setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../../")
source("scripts/windturbines/00_config.R")
library(maptools)
library(sp)
library(ggspatial)
library(rnaturalearth)
library(rnaturalearthdata)

#install.packages(c("cowplot", "googleway", "ggplot2", "ggrepel", 
#                   "ggspatial", "libwgeom", "sf", "rnaturalearth", "rnaturalearthdata"))

find_turbine_lon_lats<-function(COUNTRY){
  TURBINE_LOCATIONS<-get_param(COUNTRY,
                               "PATH_RAW_IMAGES_TURBINES_MACHINE_CLASSIFIED")
  
  files<-list.files(TURBINE_LOCATIONS)
  lon_lats<-strsplit(files, "_")
  lon_lats_short<-lapply(lon_lats, head, 2)
  
  do.call(rbind, lapply(lon_lats_short, matrix, ncol=2)) %>% 
    as_tibble() %>% 
    mutate(V1 = as.numeric(V1)) %>% 
    mutate(V2 = as.numeric(V2)) %>% 
    return()
}

table_lon_lats<-find_turbine_lon_lats("FR") %>% bind_rows(find_turbine_lon_lats("CN"))

sp<-SpatialPoints(table_lon_lats, proj4string=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"))

world <- ne_countries(scale = "medium", returnclass = "sf")
st_crs(world)
st_crs(sp)

colorsERC3<-c("#c72321", "#0d8085", "#efc220")

windturbine_points <- data.frame(sp)
names(windturbine_points) <- c("Lon", "Lat")

#I found 331 Turbines!

ggplot() +
  geom_sf(data = world, color="black", fill=colorsERC3[3]) +
  coord_sf(xlim = c(-5, 140), ylim = c(20, 60), expand = FALSE) +
  geom_point(data = windturbine_points,aes(x=Lon,y=Lat), size=4, color=colorsERC3[1], shape=4, stroke=2)

### zooming in 
ggplot() +
  geom_sf(data = world, color="black", fill=colorsERC3[3]) +
  coord_sf(xlim = c(-5, 6), ylim = c(40, 52), expand = FALSE) +
  geom_point(data = windturbine_points,aes(x=Lon,y=Lat), size=4, color=colorsERC3[1], shape=4, stroke=2)

### zooming in more
ggplot() +
  geom_sf(data = world, color="black", fill=colorsERC3[3]) +
  coord_sf(xlim = c(2.1, 2.8), ylim = c(50.13, 50.5), expand = FALSE) +
  geom_point(data = windturbine_points,aes(x=Lon,y=Lat), size=4, color=colorsERC3[1], shape=4, stroke=2)

### zooming in more
ggplot() +
  geom_sf(data = world, color="black", fill=colorsERC3[3]) +
  coord_sf(xlim = c(2.74, 2.78), ylim = c(50.14, 50.16), expand = FALSE) +
  geom_point(data = windturbine_points,aes(x=Lon,y=Lat), size=4, color=colorsERC3[1], shape=4, stroke=2)



  #annotation_scale(location = "br", width_hint = 0.5) +
  
  #annotation_north_arrow(location = "br", which_north = "true", 
  #                       pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"),
  #                       style = north_arrow_fancy_orienteering) 




### run if not available
#download.file("http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip", "countries.zip")
# Then unzip
#unzip("countries.zip")
# Load maptools

# Read in the shapefile
world <- readShapeSpatial("ne_10m_admin_0_countries.shp")
proj4string(world)


# Plot France
shp <- world[world$ADMIN=="FRANCE"]
plot(shp)

plot(sp, add=TRUE)
