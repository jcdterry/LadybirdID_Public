
# For Both_generators takes metadata from its environment. 
## So just need to set that to the right size 

Gen_Meta_Train_Speedy<-function(){
  
  if(!exists('Gen_Meta_Speedy_train_counter')){Gen_Meta_Speedy_train_counter<<-1}
  
  meta_matrix<-  Pre_AccessedMetadata_train[[1]][[Gen_Meta_Speedy_train_counter]]
  class<-   Pre_AccessedMetadata_train[[2]][[Gen_Meta_Speedy_train_counter]]
  
  Gen_Meta_Speedy_train_counter <<-Gen_Meta_Speedy_train_counter+1
  
  if(Gen_Meta_Speedy_train_counter>TrainingStepsPerEpoch){
    Gen_Meta_Speedy_train_counter <<- 1}
  return(list(meta_matrix, class))
}

Gen_Meta_Valid_Speedy<-function(){
  
  if(!exists('Gen_Meta_Speedy_valid_counter')){Gen_Meta_Speedy_valid_counter<<-1}
  
  meta_matrix<-  Pre_AccessedMetadata_valid[[1]][[Gen_Meta_Speedy_valid_counter]]
  class<-   Pre_AccessedMetadata_valid[[2]][[Gen_Meta_Speedy_valid_counter]]
  
  Gen_Meta_Speedy_valid_counter <<-Gen_Meta_Speedy_valid_counter+1
  if(Gen_Meta_Speedy_valid_counter>ValidateStepsPerEpoch){
    Gen_Meta_Speedy_valid_counter <<- 1}
  return(list(meta_matrix, class))
}


Gen_Meta_Train_Speedy_P<-function(){
  Output<-Gen_Meta_Train_Speedy()
  return(list(Output[[1]][,1:3], Output[[2]]))
}
Gen_Meta_Valid_Speedy_P<-function(){
  Output<-Gen_Meta_Valid_Speedy()
  return(list(Output[[1]][,1:3], Output[[2]]))
}
Gen_Meta_Train_Speedy_S<-function(){
  Output<-Gen_Meta_Train_Speedy()
  return(list(Output[[1]][,1:47], Output[[2]]))
}
Gen_Meta_Valid_Speedy_S<-function(){
  Output<-Gen_Meta_Valid_Speedy()
  return(list(Output[[1]][,1:47], Output[[2]]))
}
Gen_Meta_Train_Speedy_SL<-function(){
 Gen_Meta_Train_Speedy()
}
Gen_Meta_Valid_Speedy_SL<-function(){
 Gen_Meta_Valid_Speedy()
}




