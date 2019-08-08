#### Function to run an entire set of final models

### Does not include generators, metadata, etc (basically all the things at the top of final model fitting script)

### Saves history as images in the named model fit folders


FitFinalModel6 <- function( lr ,  epochs ,FolderName){
  
  cat('Have you checked the layer name references are valid?')
  cat('\nHave you checked metadata is all the right size')
  
  
  CALLBACKS <- list(callback_early_stopping( monitor = "val_loss" ,
                                             patience = 4,
                                             restore_best_weights = TRUE, 
                                             verbose = 1 ,
                                             min_delta = 0.01)  ,
                    callback_reduce_lr_on_plateau(monitor = "val_loss" ,
                                                  patience = 2,
                                                  verbose = 1))
  
  META_CALLBACKS <- list(callback_early_stopping( monitor = "val_loss" ,
                                             patience = 8,
                                             restore_best_weights = TRUE, 
                                             verbose = 1 ,
                                             min_delta = 0.01)  ,
                    callback_reduce_lr_on_plateau(monitor = "val_loss" ,
                                                  patience = 4,
                                                  verbose = 1))
  
  
  dir.create(paste0('../../',FolderName), showWarnings = FALSE)
  
  # Stage 1 first fit to images
  
  if(epochs$IM1>0){
    
    K$clear_session()
    
    conv_base_IRv2 <- application_inception_resnet_v2(weights = 'imagenet',  
                                                      include_top = FALSE,
                                                      input_shape = c(299,299,3), 
                                                      pooling ='max')
    
    Image_Predictions<- conv_base_IRv2$output%>%
      layer_dropout(rate=0.3,
                    name = 'Image_Base_Dropout_1') %>%
      layer_batch_normalization(name = 'image_normalisation')%>%
      layer_dense(units = 256,
                  activation = 'relu',
                  kernel_regularizer =  regularizer_l2(l = 0.0001),
                  name = 'BaseTop1')   %>%
      layer_dropout(rate=0.3,
                    name = 'Image_Base_Dropout_2') %>%
      layer_dense(units =18,
                  activation = 'softmax',
                  name = 'main_output')
    
    Image_Model <- keras_model(inputs = conv_base_IRv2$input,
                               outputs = Image_Predictions) 
    
    freeze_weights(Image_Model, to = 'conv_7b_ac') 
    
    Image_Model %>% compile(  loss = "categorical_crossentropy",  
                              optimizer = optimizer_adam(lr = lr$IM1),
                              metrics='categorical_accuracy')
    
    Image_Fit_History <- Image_Model %>% fit_generator(  Gen_Image_Train,
                                                         steps_per_epoch = TrainingStepsPerEpoch,
                                                         epochs = epochs$IM1, 
                                                         validation_data = Gen_Image_Valid,
                                                         validation_steps = ValidateStepsPerEpoch, 
                                                         view_metrics = FALSE)
    
    save_model_hdf5(Image_Model,paste0('../../', FolderName, '/Image_Model_S1'))
    
    try({ggsave(plot(Image_Fit_History),filename = paste0('../../', FolderName, '/Image_Fit_History.png'), height = 10, width=5,units = 'in')})
    
  }
  
  ## Stage 2 - Unfreezing feature extractor and refining fit
  
  
  if(epochs$IM2>0){
    
    K$clear_session()
    
    basemodel<- load_model_hdf5(paste0('../../', FolderName, '/Image_Model_S1'))
    
    basemodel<-unfreeze_weights(basemodel, from = 'conv_7b')
    
    basemodel %>% compile(  loss = "categorical_crossentropy",
                            optimizer = optimizer_adam(lr = lr$IM2),
                            metrics = "categorical_accuracy")
    
    Image_Fit_History2 <- basemodel %>% fit_generator( Gen_Image_Train,
                                                       steps_per_epoch = TrainingStepsPerEpoch,
                                                       epochs = epochs$IM2,
                                                       validation_data = Gen_Image_Valid,
                                                       validation_steps = ValidateStepsPerEpoch, 
                                                       view_metrics = FALSE,
                                                       callbacks = CALLBACKS)
    
    save_model_hdf5(basemodel,paste0('../../', FolderName, '/Image_Model_S2'))
    
    try({ggsave(plot(Image_Fit_History2),filename = paste0('../../', FolderName, '/Image_Fit_History2.png'), height = 10, width=5,units = 'in')})
    
    
  }
  # Primary Metadata model
  if(epochs$M1>0){
    
    K$clear_session()
    
    meta_input = layer_input(shape=  3, name='metadata_input')
    
    meta_predictions<- meta_input %>%
      layer_gaussian_noise(stddev = 0.1,
                           name = 'meta_noise')%>%
     #layer_dropout(rate=0.3,name = 'meta_dropout1')%>%
      layer_batch_normalization(name= 'meta_input_normalisation')%>%
      layer_dense( units=16,
                   activation = "relu",
                   name = 'meta1',
                   kernel_regularizer = regularizer_l2(0.00001) ) %>%
      layer_dropout(0.3, name = 'meta_dropout2')%>%
      layer_dense( units = 16, 
                   kernel_regularizer =  regularizer_l2(0.00001),
                   activation = 'relu',
                   name = 'meta2' )%>%
      layer_dropout(0.3, name = 'meta_dropout3')%>%
      layer_dense(units = 18,
                  activation = 'softmax', name = 'meta_output')
    
    Meta_Model<- keras_model(inputs = meta_input, outputs = meta_predictions)
    
    Meta_Model %>% compile(  loss = "categorical_crossentropy",
                             optimizer = optimizer_adam(lr = lr$M1),
                             metrics = "categorical_accuracy")
    
    Meta_Model_Fit <-Meta_Model %>% fit_generator( generator =  Gen_Meta_Train_Speedy_P,
                                                   steps_per_epoch = TrainingStepsPerEpoch,
                                                   epochs = epochs$M1,
                                                   validation_data = Gen_Meta_Valid_Speedy_P,
                                                   validation_steps = ValidateStepsPerEpoch, 
                                                   view_metrics = FALSE,
                                                   callbacks = META_CALLBACKS)
    
    save_model_hdf5(Meta_Model,paste0('../../', FolderName, '/Meta_Model_Prim'))
    
    
    try({ggsave(plot(Meta_Model_Fit),filename = paste0('../../', FolderName, '/Meta_Model_Prim_Fit.png'), height = 10, width=5,units = 'in')})
    
    
  } 
  
  # Primary and Secondary Metadata model
  if(epochs$M2>0){
    
    K$clear_session()
    
    meta_input = layer_input(shape=  47, name='metadata_input')
    
    meta_predictions<- meta_input %>%
      layer_gaussian_noise(stddev = 0.2,
                           name = 'meta_noise')%>%
      #layer_dropout(rate=0.3,name = 'meta_dropout1')%>%
      layer_batch_normalization(name= 'meta_input_normalisation')%>%
      layer_dense( units=64,
                   activation = "relu",
                   name = 'meta1',
                   kernel_regularizer = regularizer_l2(0.0001) ) %>%
      layer_dropout(0.2, name = 'meta_dropout2')%>%
      layer_dense( units = 64, 
                   kernel_regularizer =  regularizer_l2(0.0001),
                   activation = 'relu',
                   name = 'meta2' )%>%
      layer_dropout(0.2, name = 'meta_dropout3')%>%
      layer_dense(units = 18,
                  activation = 'softmax', name = 'meta_output')
    
    Meta_Model<- keras_model(inputs = meta_input, outputs = meta_predictions)
    
    Meta_Model %>% compile(  loss = "categorical_crossentropy",
                             optimizer = optimizer_adam(lr = lr$M2),
                             metrics = "categorical_accuracy")
    
    Meta_Model_Fit <-Meta_Model %>% fit_generator( generator =  Gen_Meta_Train_Speedy_S,
                                                   steps_per_epoch = TrainingStepsPerEpoch,
                                                   epochs = epochs$M2,
                                                   validation_data = Gen_Meta_Valid_Speedy_S,
                                                   validation_steps = ValidateStepsPerEpoch, 
                                                   view_metrics = FALSE,
                                                   callbacks = META_CALLBACKS)
    
    save_model_hdf5(Meta_Model,paste0('../../', FolderName, '/Meta_Model_Secn'))
    
    
    try({ggsave(plot(Meta_Model_Fit),filename = paste0('../../', FolderName, '/Meta_Model_Secn_Fit.png'), height = 10, width=5,units = 'in')})
    
    
  }
  # Primary, Secondary and label Metadata model
  
  
  # Combined Model_With Secondary Metadata
  
  ## Stage 1 Fitting the Top
  
  if(epochs$B1>0){
    
    K$clear_session()
    
    base_image_model<-load_model_hdf5(paste0('../../', FolderName, '/Image_Model_S2'))
    base_meta_model<- load_model_hdf5(paste0('../../', FolderName, '/Meta_Model_Secn'))
    
    base_image_model_decap<-(base_image_model %>% get_layer('BaseTop1'))$output            # Always need to check these match
    base_meta_model_decap<-(base_meta_model %>% get_layer('meta2'))$output
    
    BothTogetherOutput<-list(layer_batch_normalization(base_image_model_decap,
                                                       name = 'Image_Normaliser'), 
                             layer_batch_normalization(base_meta_model_decap,
                                                       name = 'Meta_Normaliser'))%>%
      layer_concatenate( name='LinkMeta_BaseTop')%>%
      layer_dropout(rate=0.4,
                    name= 'Post-ConCat_Dropout')%>%
      layer_dense(units = 64,
                  activation = 'relu',
                  kernel_regularizer =  regularizer_l2(l = 0.001),
                  name = 'Combo1')%>%
      layer_dropout(rate=0.4,
                    name= 'Final_PostConCat_Dropout')%>%
      layer_dense(units = 18, 
                  activation = 'softmax', name = 'Combined_output')
    
    MultiInputModel_1<-keras_model(inputs = c(base_image_model$input,
                                              base_meta_model$input),
                                   outputs = BothTogetherOutput)
    
    freeze_weights(MultiInputModel_1, to='LinkMeta_BaseTop') # Freeze all lower parts
    
    MultiInputModel_1 %>% compile(  loss = "categorical_crossentropy", 
                                    optimizer =  optimizer_adam(lr = lr$B1),
                                    metrics='categorical_accuracy')
    
    Both_Fit_History_1 <-MultiInputModel_1 %>% fit_generator( generator =  Gen_Both_Train_Secn,
                                                              steps_per_epoch = TrainingStepsPerEpoch,
                                                              epochs = epochs$B1,
                                                              validation_data = Gen_Both_Valid_Secn,
                                                              validation_steps = ValidateStepsPerEpoch, 
                                                              view_metrics = FALSE)
    
    save_model_hdf5(MultiInputModel_1, paste0('../../', FolderName, '/Both_Model_S1'))
    
    try({ggsave(plot(Both_Fit_History_1),filename = paste0('../../', FolderName, '/Both_Fit_History_1.png'), height = 10, width=5,units = 'in')})
    
    
  }
  
  ## Final Refining
  
  if(epochs$B2>0){
    
    
    K$clear_session()
    
    MultiInputModel_2<- load_model_hdf5(paste0('../../', FolderName, '/Both_Model_S1'))
    MultiInputModel_2<-unfreeze_weights(MultiInputModel_2, from = 'meta1')
    MultiInputModel_2<-unfreeze_weights(MultiInputModel_2, from = 'BaseTop1')
    
    MultiInputModel_2 %>% compile(  loss = "categorical_crossentropy",
                                    optimizer = optimizer_adam(lr = lr$B1),
                                    metrics = "categorical_accuracy")
    
    Both_Fit_History_2 <- MultiInputModel_2 %>% fit_generator( Gen_Both_Train_Secn,
                                                               steps_per_epoch = TrainingStepsPerEpoch,
                                                               epochs = epochs$B2,
                                                               validation_data = Gen_Both_Valid_Secn,
                                                               validation_steps = ValidateStepsPerEpoch, 
                                                               view_metrics = FALSE,
                                                               callbacks = CALLBACKS)
    
    save_model_hdf5(MultiInputModel_2,paste0('../../', FolderName, '/Both_Model_S2'))
    try({ggsave(plot(Both_Fit_History_2),
                filename = paste0('../../', FolderName, '/Both_Fit_History_2.png'), height = 10, width=5,units = 'in')})
    
    
  }

}


