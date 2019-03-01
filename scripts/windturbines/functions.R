library(rjson)
library(stringr)
library(feather)
library(tidyr)
library(tidyverse)
library(sp)
library(rgdal)
library(raster)
library(rgeos)
library(dismo)
library(data.table)
library(geosphere)
library(foreign)
library(sf)
library(parallel)


###imports windturbine data from ig-windkraft, adds column with windpark-name,
###corrects wrong data points, and saves results to filename in feather format
importAndSaveIGWindData<-function(fileName){
  
  # import IG-Wind data
  json_file <- "https://www.igwindkraft.at/src_project/external/maps/generated/gmaps_daten.js"
  lines <- readLines(json_file)
  lines[1] <- sub(".* = (.*)", "\\1", lines[1])
  lines[length(lines)] <- sub(";", "", lines[length(lines)])
  json_data <- fromJSON(paste(lines, collapse="\n"))
  
  
  # extract data into data frame
  col_names <- c("Name","Betreiber1","Betreiber2","n_Anlagen","KW","Type","Jahr","x","Lat","Long","url","Hersteller","Nabenhoehe","Rotordurchmesser")
  wind_turbines_data        <- data.frame(matrix(ncol = 14, nrow = 0))
  names(wind_turbines_data) <- col_names
  
  
  for (i in seq(json_data[[2]]$places)){
    for(j in seq(col_names)){
      if (!is.null(json_data[[2]]$places[[i]]$data[[j]])){
        wind_turbines_data[i,j] <- json_data[[2]]$places[[i]]$data[[j]]
      } else {
        wind_turbines_data[i,j] <- NA 
      }
    }
  }
  
  wind_turbines_data$x <- NULL
  wind_turbines_data <- wind_turbines_data %>% mutate(Name_Save=Name)
  name<-wind_turbines_data$Name
  name<-str_replace(name,"ä","ae")
  name<-str_replace(name,"ö","oe")
  name<-str_replace(name,"ü","ue")
  name<-str_replace(name,"ß","sz")
  name<-str_replace(name," ","_")
  name<-str_replace(name,"\\(","-")
  name<-str_replace(name,"\\)","-")
  
  wind_turbines<-wind_turbines_data
  wind_turbines$Name<-name
  
  wind_turbines<-wind_turbines %>% mutate(Park=matrix(unlist(strsplit(wind_turbines$Name,",")), 
                                                      ncol=2, 
                                                      byrow=TRUE)[,1]) %>% mutate(Park=str_replace(Park,"/",""))
  
  wind_turbines<-remove_erroneous_data(wind_turbines)
  
  wind_turbines<-wind_turbines %>% mutate(Nabenhoehe=as.numeric(Nabenhoehe),Rotordurchmesser=as.numeric(Rotordurchmesser))
  
  ###remove long/lat duplicates
  longlat<-windTurbines %>% mutate(n=1:n()) %>% dplyr::select(Long,Lat,n)
  windTurbines_filtered<-windTurbines %>% 
    filter(!(row_number() %in% longlat[duplicated(longlat[,1:2]),]$n)) 
  
  
  write_csv(wind_turbines,fileName)
  

  return(wind_turbines)
}

###remove / correct wrong data points in original file
remove_erroneous_data<-function(wind_turbines){
  
  ##zurndorf: Lat/Long vertauscht
  ##zurndorf: Lat/Long vertauscht
  selector<-wind_turbines$Lat<40
  helper<-wind_turbines[selector,]$Lat
  wind_turbines[selector,]$Lat<-wind_turbines[selector,]$Long
  wind_turbines[selector,]$Long<-helper
  
  ##nickelsdorf anlage 9: falsch
  selector<-wind_turbines$Long>17.39
  wind_turbines[selector,]$Long<-17+(wind_turbines[selector,]$Long-17)/10
  
  ##anlagen munderfing 5 und 6 eigenartig
  wind_turbines[wind_turbines$Park=="Munderfing"&wind_turbines$Long>14,]$Long<-
    wind_turbines[wind_turbines$Park=="Munderfing"&wind_turbines$Long>14,]$Long-3.5
  
  ##Correct very large latitude values (e.g. Pottendorf)
  wind_turbines[wind_turbines$Lat>1000,]$Lat<-wind_turbines[wind_turbines$Lat>1000,]$Lat/10^5
  
  ##Gro?hofen: one turbine far away from others
  wind_turbines[wind_turbines$Park=="Groszhofen"&wind_turbines$Long<16.6,]$Long<-
    wind_turbines[wind_turbines$Park=="Groszhofen"&wind_turbines$Long<16.6,]$Long+0.52
  
  return(wind_turbines)
}

###calculates the convex hull of a wind park and writes the corresponding
###shape file to disk
###input: long and lat coordinates of all turbines in the park and the parkname
###as sideeffect a new shapefile with the name of the windpark is created in the subdirectory
###GIS
calculateHull<-function(Long,Lat,Park,directory){
  
  print(paste(Park[1],
              "with",
              length(Long),
              "Turbines"))
  coords<-NULL
  
  
  dat<-data.frame(Long,Lat)
  
  
  if(length(Long)<3){
    if(length(Long)==1){
      coords<-data.frame(Long=c(Long[1]-0.005,
                                Long[1]+0.005,
                                Long[1]+0.005,
                                Long[1]-0.005,
                                Long[1]-0.005),
                         
                         Lat=c(Lat[1]-0.005,
                               Lat[1]-0.005,
                               Lat[1]+0.005,
                               Lat[1]+0.005,
                               Lat[1]-0.005))
    }else{
      if(Long[1]==Long[2]){
        Long[1]<-Long[1]-0.01
      }
      if(Lat[1]==Lat[2]){
        Lat[1]<-Lat[1]-0.01
      }
      
      
      coords<-data.frame(Long=c(Long[1],
                                Long[2],
                                Long[2],
                                Long[1],
                                Long[1]),
                         Lat=c(Lat[1],
                               Lat[1],
                               Lat[2],
                               Lat[2],
                               Lat[1]))
      
    }
    
    
  }else{
    
    
    
    ch <- chull(dat)
    coords <- dat[c(ch, ch[1]), ]  # closed polygon
    
  }
  
  
  #pdf(paste("data/pdfsConvexHull/",Park[1],".pdf",sep=""))
  #plot(dat, pch=19)
  #lines(coords, col="red")
  #dev.off()
  
  
  sp_poly <- SpatialPolygons(list(Polygons(list(Polygon(coords)), ID=1)),proj4string=CRS("+proj=longlat +datum=WGS84"))
  # set coordinate reference system with SpatialPolygons(..., proj4string=CRS(...))
  # e.g. CRS("+proj=longlat +datum=WGS84")
  sp_poly_df <- SpatialPolygonsDataFrame(sp_poly, data=data.frame(ID=1))
  setwd(getwd())
  writeOGR(sp_poly_df, paste0(directory,Park[1],".shp"), layer=Park[1], driver="ESRI Shapefile",overwrite_layer=TRUE)
  return(0)
}


shapeAllTurbines<-function(Long,Lat,Park,directory){
  
  print(paste(Park[1],
              "with",
              length(Long),
              "Turbines"))
  
  sub_union<-NULL
  
  for(i in 1:length(Long)){
    square<-data.frame(x=c(Long[i]-0.002,
                           Long[i]-0.002,
                           Long[i]+0.002,
                           Long[i]+0.002),
                        y=c(Lat[i]+0.002,
                            Lat[i]-0.002,
                            Lat[i]-0.002,
                            Lat[i]+0.002))
    
    
    sp_poly <- SpatialPolygons(list(Polygons(list(Polygon(square)), ID=1)),proj4string=CRS("+proj=longlat +datum=WGS84"))
    # set coordinate reference system with SpatialPolygons(..., proj4string=CRS(...))
    # e.g. CRS("+proj=longlat +datum=WGS84")
    sp_poly_df <- SpatialPolygonsDataFrame(sp_poly, data=data.frame(ID=1))
    if(is.null(sub_union)){
      sub_union<-sp_poly_df
    }else{
      names(sp_poly_df)=paste0("ID",i)
      sub_union<-union(sub_union,sp_poly_df)
    }
    
    
    
    
  }
 
  
  #pdf(paste("data/pdfsConvexHull/",Park[1],".pdf",sep=""))
  #plot(dat, pch=19)
  #lines(coords, col="red")
  #dev.off()
  setwd(getwd())
  writeOGR(sub_union, paste0(directory,Park[1],".shp"), layer=Park[1], driver="ESRI Shapefile",overwrite_layer=TRUE)
  
  
  
  return(0)
}

tile2Coords<-function(x,y,z){
  

  
  n<-pi-2*pi*y/(2^z)
  lat2<-(180/pi*atan(0.5*(exp(n)-exp(-n))))

  
  n<-pi-2*pi*(y+1)/(2^z)
  lat1<-(180/pi*atan(0.5*(exp(n)-exp(-n))))
  

  long1<-(x/(2^z)*360-180)
  long2<-((x+1)/(2^z)*360-180)
  
  return(c(long1,long2,lat1,lat2))  
  
}


save_tile<-function(file,lon,lat,zoom){
  
  #gettile long lat
  TILE_SIZE <- 256
  scale<-bitwShiftL(1,zoom)
  
  
  
  siny<- sin(lat * pi / 180)
  
  siny<- min(max(siny, -0.9999), 0.9999)
  
  x<-TILE_SIZE * (0.5 + lon / 360)
  y<-TILE_SIZE * (0.5 - log((1 + siny) / (1 - siny)) / (4 * pi))
  
  #print(x)
  #print(y)
  
  #print(x*scale)
  #print(y*scale)
  
  cols<-floor(x * scale / TILE_SIZE)
  rows<-floor(y * scale / TILE_SIZE)
  
  
  #url<-paste0("https://maps1.wien.gv.at/basemap/bmaporthofoto30cm/normal/google3857/",zoom,"/",rows,"/",cols,".jpg")
  cnt<-1
  
  #layout(matrix(1:9, ncol=3, byrow=TRUE))
  
  temp<-PATH_LOCAL_TEMP
  dir.create(temp, showWarnings = FALSE)
  
  prefix<-floor(runif(1)*100000) 
  
  for(i in -1:1){
    for(j in -1:1){
      #print(i)
      url<-paste0("http://mt0.google.com/vt/lyrs=s&hl=en&x=",cols+i,"&y=",rows+j,"&z=",zoom,"")
      #print(url)
      f<-paste0(temp,prefix,cnt,".jpg")
      download.file(url, f, mode = "wb",quiet=TRUE)
     
      
      img<-brick(f)
      coords<-tile2Coords(cols+i,rows+j,zoom)
      #print(coords)
      xmin(img) <- coords[1] 
      xmax(img) <- coords[2] 
      ymin(img) <- coords[3]
      ymax(img) <- coords[4]
      crs(img) <- CRS("+init=epsg:4326")
      f<-paste0(temp,prefix,cnt,".tif")
      writeRaster(img,f)
      
      cnt<-cnt+1

      }
      
  }
  
  files<-paste0(temp,prefix,1:9,".tif")
  all_raster<-sapply(files,brick)
  fin<-all_raster[[1]]
  for(i in 2:9){
    fin<-merge(fin,all_raster[[i]],tolerance=1)
  }

  files_jpg<-paste0(temp,prefix,1:9,".jpg")
  
  unlink(files)
  unlink(files_jpg)
  
  
  
    
  
  
  s <- brick(nl=3,nrow = 256, ncol = 256)
  extent(s) <- extent(fin)
  ras <- resample(fin, s, method = 'bilinear')
  
  #plotRGB(ras)
  
  #print(file)
  #print(ras)
  writeRaster(ras,file)
  
}
  
  
  



createNonWindTurbineImagesRandom<-function(windTurbines_filtered,
                                           n,directory,shapeLoc,zoom=17){
  
  dir.create(file.path(".", directory), 
             showWarnings = FALSE)
  
  filename<-paste0(shapeLoc,"union_parks.shp")
  ###write shape files to disk
   subs_union<-NULL
  if(file.exists(filename)){
    subs_union <- readOGR(filename)
    
    
  }else{
    windTurbines_filtered %>% group_by(Park) %>% dplyr::summarize(s=shapeAllTurbines(Long,Lat,Park,PATH_WINDPARK_LOCATIONS))
    files<-list.files(path = shapeLoc,pattern="shp")
    
    subs_union<-readOGR(paste0(shapeLoc,files[1]))
    for(i in 1:length(files)){
      print(i)
      subs <- readOGR(paste0(shapeLoc,files[i]))
      names(subs)=paste0("ID",i)
      subs_union<-union(subs_union,subs)
    
    
    }
    #subs_union<-spTransform(subs_union,projection)  
    writeOGR(subs_union, paste0(shapeLoc,"union_parks.shp"), layer="union_parks", driver="ESRI Shapefile",overwrite_layer=TRUE)
    
    
  }
  
   
  

  minX<-min(windTurbines_filtered$Long)
  maxX<-max(windTurbines_filtered$Long)
  minY<-min(windTurbines_filtered$Lat)
  maxY<-max(windTurbines_filtered$Lat)
  
  
  logfile<-logfile<-paste0(PATH_TEMP,"out_NoWindturbines.log")
  
  unlink(logfile)
  cl <- makeCluster(4,
                    outfile=logfile)
  
  clusterEvalQ(cl, source("scripts/windturbines/functions.R"))
  clusterEvalQ(cl, source("scripts/windturbines/00_config.R"))
  
  print(paste0("Running in parallel mode. Check ",logfile," for subprocess outputs."))
  
  parLapply(cl,
            1:n,
            getOneRandomTileNoTurbine,
            directory,
            subs_union,
            minX,
            maxX,
            minY,
            maxY,
            zoom
            )
  # one or more parLapply calls
  stopCluster(cl)
  
  
 
  
  
}

getOneRandomTileNoTurbine<-function(i,directory,subs_union,minX,maxX,minY,maxY,zoom){
    print(i)
  
    got<-TRUE
    
    while(got){
  
      x<-minX+runif(1)*(maxX-minX)
      y<-minY+runif(1)*(maxY-minY)
    
      points<-data.frame(Long=c(x),Lat=c(y))
      coordinates(points) <- c("Long", "Lat")
      proj4string(points) <- CRS("+proj=longlat +datum=WGS84")  
      points<-as(points, "SpatialPoints")
      points<-spTransform(points,projection(subs_union))
    
      points_buffer<-gBuffer(points[1,],width=160,capStyle="SQUARE")
    
    if(!gIntersects(points_buffer,subs_union)){
      
      result <- tryCatch({
        
        
        save_tile(paste0(directory,i,".tif"),x,y,zoom)
        
        got<-FALSE
      
        }, error = function(err) {
        
        print(err) 
        # return(0)
        
    }) 
      
      
      
    }
  }
}

createWindTurbineImages<-function(windTurbines_filtered,directory,zoom=17,start=1){
  
  dir.create(file.path(".", directory), 
             showWarnings = FALSE)
  
  logfile<-paste0(PATH_TEMP,"out_windturbines.log")
  
  unlink(logfile)
  
  cl <- makeCluster(4,
                   outfile=logfile)
  
  clusterEvalQ(cl, source("scripts/windturbines/functions.R"))
  clusterEvalQ(cl, source("scripts/windturbines/00_config.R"))
  
  print(paste0("Running in parallel mode. Check ",logfile," for subprocess outputs."))
  parLapply(
            cl,
            start:nrow(windTurbines_filtered),
            function(i,windTurbines_filtered,directory,zoom){
              
              fname<-paste0(directory,i,".tif")
              
              print(i)
              
              if(!file.exists(fname)){
                x<-windTurbines_filtered$Long[i]
                y<-windTurbines_filtered$Lat[i]
                result <- tryCatch({
                  
                         save_tile(fname,x,y,zoom)
                }, error = function(err) {
                  
                  print(err) 
                  # return(0)
                  
                })
              }
              },
            windTurbines_filtered,
            directory,
            zoom
  )
  # one or more parLapply calls
  stopCluster(cl)
  
}