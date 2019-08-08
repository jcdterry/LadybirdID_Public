# LadybirdID_Public

 Supporting Information for *Thinking like a naturalist: enhancing computer vision of citizen science images by harnessing contextual data* (Terry, Roy & August)
This repository contains the key parts of the code used to build and run keras models in R, and to analyse the results. 
There are four scripts:

## Fit and Analyse Models.rmd

This is the 'master' script that repeatedly fits the models onto different splits of train, validate, test data, then produces the figures used in the paper. An markdown document incorporating the outputs is also included, including full session information. It is quite long and not very readable. The most likely functions of interest are presented separately. 

## FinalModelMaker6.R 

This contains a single large function that sequentially builds and fits each of the models

## Gen_Image_TVT.R

Generator functions to supply the model with data from an external repository. Includes the image augmentation hyperparamters that were used. 

## Gen_Meta_All_PSL.R

'Speedy' generator functions for the meta-data only models that pre-access the data, as it is small enough to all fit in RAM.

Unfortunately licensing prevents us making freely available the underlying data sources, so these scripts are not runnable as such. They are intended to give the most thorough view possible of the structures and hyperparameters we used, and to provide a possible template for future work. However, if anybody did have an interest in using the data, then do get in contact!

## Version numbers used:

Tensorflow: '1.13.0rc0-gpu'

Nvidia Driver: 418.40.04

Cuda: 10.1

Python: 2.7, in r-tensorflow virtual environment

R: 3.5.3

keras R package: 2.2.4

Operating System: Ubuntu: 16.04.6
