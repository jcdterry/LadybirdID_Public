
PreProcess <- function(x){((x/255)-0.5)*2}

Gen_Image_Train_inner <- flow_images_from_directory(
  paste0(IMAGE_SOURCE,'train/'), 
  image_data_generator(rotation_range =45 , 
                       width_shift_range =0.3,
                       height_shift_range=0.3,
                       shear_range =0.2,
                       zoom_range = 0.2,
                       brightness_range = c(0.5, 1.5),
                       horizontal_flip = TRUE,
                       fill_mode = "wrap" ),
  target_size = c(299, 299),
  batch_size = 32, 
  class_mode = "categorical" )


Gen_Image_Train <- function(){
  UnScaled<-reticulate::iter_next(Gen_Image_Train_inner)
  Imgs<-UnScaled[[1]]
  for(B in 1:32){
    Imgs[B,,,] <- PreProcess(Imgs[B,,,] )
  }
  return(list(Imgs,UnScaled[[2]]))
}

Gen_Image_Valid_inner <- flow_images_from_directory(
  paste0(IMAGE_SOURCE,'validation/'), 
  image_data_generator(), 
  target_size = c(299, 299),
  batch_size = 32, 
  class_mode = "categorical" )


Gen_Image_Valid <- function(){
  UnScaled<-reticulate::iter_next(Gen_Image_Valid_inner)
  Imgs<-UnScaled[[1]]
  for(B in 1:32){
    Imgs[B,,,] <- PreProcess(Imgs[B,,,] )
  }
  return(list(Imgs,UnScaled[[2]]))
}


Gen_Both_Train <- function() {
  
  B_i<-Gen_Image_Train_inner$batch_index+1   #  which ones to draw:(do before iteration)
  B_size <-Gen_Image_Train_inner$batch_size
  
  x1 <- reticulate::iter_next(Gen_Image_Train_inner) # Generate the data
  
  for(B in 1:BatchSize){
    x1[[1]][B,,,] <- PreProcess(x1[[1]][B,,,] )
  }
  
  IndexOrder<-Gen_Image_Train_inner$index_array 
  FileNames<- Gen_Image_Train_inner$filenames[IndexOrder+1]
  Batch_numbers<- (B_size*(B_i-1)+1):(B_size*(B_i))
  
  str_extract(FileNames[Batch_numbers],                      ## Get names of the files used in batch. 
              pattern = '(?<=_)[:digit:]+(?=_|.)') %>% # any num preceded by _ and followed by either _ or .
    as.numeric%>%tibble(id = .)%>%
    left_join(metadata, by = "id")%>%   # Join to metadata table from global environment
    select(-id)%>%                      # clean up for keras
    as.matrix-> metadata_matrix         ## output array (matrix)
  dimnames(metadata_matrix)<-NULL
  
  list (list(x1[[1]], metadata_matrix), x1[[2]]) 
}

Gen_Both_Valid <- function() {
  
  B_i<-Gen_Image_Valid_inner$batch_index    +1                        
  x1 <- reticulate::iter_next(Gen_Image_Valid_inner) # Generate the data
  
  for(B in 1:BatchSize){
    x1[[1]][B,,,] <- PreProcess(x1[[1]][B,,,] )
  }
  
  IndexOrder<-Gen_Image_Valid_inner$index_array+1
  FileNames<- Gen_Image_Valid_inner$filenames[IndexOrder]
  
  B_size <-Gen_Image_Valid_inner$batch_size
  Batch_numbers<-  (B_size*(B_i-1)+1):(B_size*(B_i))
  
  str_extract(FileNames[Batch_numbers]  ,     ## Get names of the files used in batch. 
              pattern = '(?<=_)[:digit:]+(?=_|.)') %>% # num preceded by _ and followed by either _ or .
    as.numeric%>%tibble(id = .)%>%
    left_join(metadata, by = "id")%>%   # Join to metadata table from global environment
    select(-id)%>%                      # clean up for keras
    as.matrix-> metadata_matrix         ## output array (matrix)
  dimnames(metadata_matrix)<-NULL
  
  list (list(x1[[1]], metadata_matrix), x1[[2]]) 
}


Gen_Both_Train_Secn <- function(){
  output<- Gen_Both_Train()
  output[[1]][[2]] <-  output[[1]][[2]][,1:L_Meta_Secn] # Trim Metadata to size
  return(output)
}

Gen_Both_Valid_Secn <- function(){
  output<- Gen_Both_Valid()
  output[[1]][[2]] <-  output[[1]][[2]][,1:L_Meta_Secn] # Trim Metadata to size
  return(output)
}

