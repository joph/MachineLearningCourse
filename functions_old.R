source("libraries.R")


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
  name<-str_replace(name,"?","ae")
  name<-str_replace(name,"?","oe")
  name<-str_replace(name,"?","ue")
  name<-str_replace(name,"?","sz")
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

  write_feather(wind_turbines,fileName)
  
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
calculateHull<-function(Long,Lat,Park){
  
  print(paste(Park[1],"with",length(Long),"Turbines"))
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

  
  pdf(paste("results/pdfsConvexHull/",Park[1],".pdf",sep=""))
  plot(dat, pch=19)
  lines(coords, col="red")
  dev.off()
  
  
  sp_poly <- SpatialPolygons(list(Polygons(list(Polygon(coords)), ID=1)),proj4string=CRS("+proj=longlat +datum=WGS84"))
  # set coordinate reference system with SpatialPolygons(..., proj4string=CRS(...))
  # e.g. CRS("+proj=longlat +datum=WGS84")
  sp_poly_df <- SpatialPolygonsDataFrame(sp_poly, data=data.frame(ID=1))
  writeOGR(sp_poly_df, "GIS", layer=Park[1], driver="ESRI Shapefile",overwrite_layer=TRUE)
  return(0)
}

###Writes the input file necessary to run
###the gams optimization to disk
###in particular writes a matrix with distances,
###and the productivity of the locations,
###dir indicates the output directory to put the gdx files
writeOptmodelDistFiles<-function(Long,Lat,name,dir){
  pp<-tibble(Long,Lat)
  
  if(nrow(pp)==1){
     pp<-bind_rows(pp,pp)  
  }
  
  print(paste("Writing ",name, " --------------------------------------------"))
  
  print(pp)
  
  
    distMatrix<-mapply(combDist,pp$Long,pp$Lat,list(pp))
    diag(distMatrix)<-10000

  #print(distMatrix)
  
  
  #limitMatrix<-distMatrix*0
  #limitMatrix[distMatrix<dist]<-1
  
  distMatrix<-as_tibble(distMatrix)
  ns<-paste("i",formatC(1:ncol(distMatrix),2,format="d",flag="0"),sep="")
  names(distMatrix)<-ns
  distMatrix<-bind_cols(tibble(ns),distMatrix) %>% gather(variable,val,-ns) 
  distMatrix<-distMatrix[order(nchar(distMatrix$ns), distMatrix$ns,nchar(distMatrix$variable),distMatrix$ns),]
  
 gdxList<-list(tidyToGDXparam("distMatrix",
                               distMatrix,
                               uelColumns=1:2,
                               valColumn="val")
              
  )
  
  
  wgdx.lst(paste("GDX/",dir,"/",name,".gdx",sep=""),gdxList)
  
  #write_delim(limitMatrix,paste("CSV/",name,".csv",sep=""),delim=";")
  return(0)
}

###Creates, from a time series of windspeeds, an annual generation of power output
###input values are the a and k parameter and
###parameters of the wind turbine
###as a result, the generation in MWh is returned
transformToPowerOutput<-function(a,k,i,cut_in=2,cut_out=25,hubheight=100,diameter=82,capacity=2)
{
  if((i %% 100) == 0){
    
    print(paste(i,"simulations done"))
  }
  #print(paste("a:",a,"k:",k))
  windspeeds<-rweibull(8760,
                       k,
                       a)
  
  roughness<-0.04
  windSpeeds<-windspeeds*log(hubheight/roughness)/log(100/roughness)
  
  air_density  <-  1.16375
  # Berechnung des power outputs
  windSpeeds<-ifelse(windSpeeds<=cut_in,0,windSpeeds)
  #Cut-out at 25 m/s
  windSpeeds<-ifelse(windSpeeds>cut_out,0,windSpeeds)
  weib3<-windSpeeds^3
  ###0.40: average efficiency of wind turbine
  ###0.88: further losses of 12%
  Pow_Out <-1/2*0.40*0.88*air_density*pi*(diameter/2)^2*weib3/10^3 #Umrechnung Watt in kW

  maxPow_Out<-ifelse(Pow_Out>=capacity,capacity,Pow_Out)
  
  #aggregation and conversion to MW
  return(sum(maxPow_Out)/1000)
}


####simulate windpower production at all points
simAtAllPoints<-function(all_data,
                         newCap,
                         newHHeight,
                         newDiam){
  
  all_data[all_data$Type=="additionalPoints"|all_data$Type=="originalPoints",5:7]<-data.frame(c(newCap),c(newHHeight),c(newDiam))
  
  
  ###simulate wind speeds
  ###extrapolate to hub height
  ###simulate wind power production
  prod<-mapply(transformToPowerOutput,
               all_data$a_extract,
               all_data$k_extract,
               1:nrow(all_data),
               2,
               25,
               all_data$hubheight,
               all_data$diameter,
               all_data$KW)
  all_data$annualProduction<-prod
  return(all_data)
}



####simulate windpower production at all points
####and save to gdx file
writeOptmodelProductivityFile<-function(all_data_in,name,dir){
  all_data<-all_data_in %>% filter(all_data_in$Park==name)
  productivity<-tibble(paste("i", formatC(1:nrow(all_data),2,format="d",flag="0"),sep=""),
                     all_data$annualProduction)
  names(productivity)<-c("i","val")

  gdxList<-list(tidyToGDXparam("productivity",
               productivity,
               uelColumns=1,
               valColumn="val"))
  wgdx.lst(paste("GDX/",dir,"/",name,"_productivity.gdx",sep=""),gdxList)
  return(0)
}





writeDistance<-function(dist){
  distP<-tibble(ind=c("P1"),val=c(dist))
  gdxList<-list(
                tidyToGDXparam("dist",
                               distP,
                               uelColumns=1,
                               valColumn="val")
  )
  
  
  wgdx.lst("GAMS/input1.gdx",gdxList)
}




#####calculates points where wind turbines may be installed from the incoming shape file
convertShapeToPoints<-function(shapeFile,gridResolution){
  
  # Load our shapefile. Just choose the ".shp" file inside the folder you downloaded.
  shape <- readOGR(shapeFile)
  
  # Plot the shape to see if everything is fine.
  plot(shape)
  
  # Create an empty raster.
  grid <- raster(extent(shape))
  
  # Choose its resolution. I will use 2.5 degrees of latitude and longitude.
  res(grid) <- gridResolution
  
  # Make the grid have the same coordinate reference system (CRS) as the shapefile.
  proj4string(grid)<-proj4string(shape)
  
  # Transform this raster into a polygon and you will have a grid, but without Brazil (or your own shapefile).
  gridpolygon <- rasterToPolygons(grid)
  
  plot(gridpolygon)
  
  # Use an equal-area projection (which is able to preserve area measures) to be able to calculate area sizes. I will use Lambert Azimuthal Equal-Area projection.
  drylandproj<-spTransform(shape, CRS("+proj=laea"))
  gridpolproj<-spTransform(gridpolygon, CRS("+proj=laea"))
  
  # Identify each cell in our grid (use numbers as references; for the next steps).
  gridpolproj$layer <- c(1:length(gridpolproj$layer))
  
  # Calculate the area of each cell in our grid.
  areagrid <- gArea(gridpolproj, byid=T)
  
  # Intersect the grid with the shape. This can take some time.
  dry.grid <- intersect(drylandproj, gridpolproj)
  
  # Calculate the area of each cell in our intersected grid.
  areadrygrid <- gArea(dry.grid, byid=T)
  
  # Identify each cell, its area in the original grid, and its area in the intersected grid.
  info <- cbind(dry.grid$layer, areagrid[dry.grid$layer], areadrygrid)
  
  # Divide the area of each cell in the intersected grid by the area of the same cell in our original grid (i.e. before intersection). Then, save these proportions as the attribute "layer".
  dry.grid$layer<-info[,3]/info[,2]
  
  # Make the gridded shape have WGS84 projection (default projection in most GIS softwares).
  dry.grid <- spTransform(dry.grid, CRS(proj4string(shape)))
  
  # Keep grid cells that have at least a specific proportion of the area covered by the shape (covered by Brazil, in our example). Let only maintain cells with at least 30% of their area covered by "Brazil". It could be area coverage of agricultural zones, of dry land, of a factory, etc.
  dry.grid.filtered <- dry.grid[dry.grid$layer >= 0.01,]
  
  xy<-coordinates(dry.grid.filtered)
  plot(dry.grid.filtered)
  points(xy,col="red")
  
  print(nrow(xy))
  
  return(xy)
  #writeOGR(dry.grid.filtered, dsn="GIS_converted", layer="final_shape", driver="ESRI Shapefile", overwrite_layer=T)
}

###gets the results of one single run and plots it
getResultsPlot<-function(dir,Park1,c_p_l,out,figure=TRUE){
  
  symbol<-rgdx(paste("GDX/",dir,"/",Park1,"_out.gdx",sep=""),request_list<-list(name="x_choose"),squeeze=FALSE)
  
  pp<-c_p_l %>% dplyr::filter(Park==Park1) 
  #%>% dplyr::select(Long,Lat)
  opt<-bind_cols(tibble(V2=symbol$val[1:nrow(pp),2]),pp) %>% mutate(Type="Opt",Park=Park1) %>% filter(V2==1) 
  #%>% 
  #  dplyr::select(Long,Lat,Park,Type)
  
  alldat<-bind_rows(out,
                    opt)
  
  if(figure){
    alldat %>% filter(Park==Park1) %>% ggplot(aes(x=Long,y=Lat))+geom_point(aes(col=Type,shape=Type))
    ggsave(paste("results/",dir,"/",Park1,".png",sep=""))
  }
  return(alldat)
}  

###reads a single symbol froma  gdx file
readSingleSymbolGDX<-function(symbol){
  name<-symbol$name
  if(symbol$dim==0){
    
    return(tibble(name=name,value=symbol$val))
  }
  
  ###construct data.frame
  out<-tibble(rep(symbol$name,nrow(symbol$val)))
  for(i in 1:(symbol$dim)){
    
    out<-bind_cols(out,tibble(symbol$uels[[i]][symbol$val[,i]]))
  }
  out<-bind_cols(out,tibble(symbol$val[,i+1]))
  names(out)<-c("name",symbol$domains,"value")
  
  
  return(out)
  
}


###copy correct input.gdx file to gams directory
###run gams and copy output back to seperate gdx file
copyAndRun<-function(Park_,all_data,dir,dist){
  print(paste("Running ",Park_))
  writeDistance(dist)
  writeOptmodelProductivityFile(all_data,Park_,dir)
  file.copy(paste("GDX/",dir,"/",Park_,".gdx",sep=""),"GAMS/input.gdx",overwrite=TRUE)
  file.copy(paste("GDX/",dir,"/",Park_,"_productivity.gdx",sep=""),"GAMS/input2.gdx",overwrite=TRUE)
  wd<-getwd()
  setwd("GAMS/")
  gams("repower_I.0.gms")
  setwd(wd)
  file.copy("GAMS/output.gdx",paste("GDX/",dir,"/",Park_,"_out.gdx",sep=""),overwrite=TRUE)
}


###helper function to apply distance function to all
###points
combDist<-function(long1,lat1,pp){
  mapply(myDist,long1,lat1,pp$Long,pp$Lat)
}


###create GAMS Parameter
###name: name of parameter
###value: data.frame or matrix with values for parameter
###uels: list of uel lists
##dim: dimension of parameter
createPARAM<-function(name,value,uels,dim=1){
  
  val<-as.matrix(value)
  return(list(name=name,type="parameter",dim=dim,form="sparse",uels=uels,val=val))
  
}

###converts a tidy data.frame to a GDX file
tidyToGDXparam<-function(name,tb,uelColumns,valColumn){
  ###uels->to numeric
  start<-sapply(tb[,uelColumns],
                function(x){
                  as.numeric(factor(unlist(x),
                                    levels=unique(sort(unlist(x)))
                  ))
                }
  )
  
  leadingIndexUEL<-as_tibble(start)
  value<-bind_cols(leadingIndexUEL,tibble(tb[[valColumn]]))
  
  
  ###generate uels
  uels_<-sapply(tb[,uelColumns],function(x){as.character(sort(unique(x)))},simplify=FALSE)
  
  ###create param
  return(createPARAM(name,value,uels_,dim=length(uelColumns)))
  
}


####calculates the distance between two long/lat points
myDist<-function(long1,lat1,long2,lat2){
  
  d<-distVincentyEllipsoid(c(long1,lat1), c(long2,lat2))
  return(d/1000)
  
}

runScenario<-function(scenario,dist,all_data_in,figure=TRUE){
  ###run all files GRID
  locs<-all_data_in %>% 
    filter(Type==scenario) %>% 
    dplyr::select(Park) %>% 
    unique() %>% 
    unlist() %>% 
    as.vector() 
  
  sapply(locs, 
         copyAndRun,
         dir=scenario,
         dist=dist,
         all_data=all_data_in)
  
  ###read results
  
  
  ad1<-all_data_in %>% filter(Type==scenario)
  out<-all_data_in 
  for(name in sort(unique(all_data_in$Park))){
    print(name)
    out<-getResultsPlot(scenario,name,ad1,out,figure)
  }
  return(out)
  
}




