#!/bin/sh

name_dir_in=./images_input/

################################################################################################

name_dir_model=./trained_models_SAH/model_cistern.pt  
name_dir_out=./results_cistern

python predict_vunet.py --outpath $name_dir_out --modelname $name_dir_model --imagepath $name_dir_in

################################################################################################

name_dir_model=./trained_models_SAH/model_sulcal.pt  
name_dir_out=./results_sulcal

python predict_vunet.py --outpath $name_dir_out --modelname $name_dir_model --imagepath $name_dir_in

################################################################################################

name_dir_model=./trained_models_SAH/model_ventri.pt  
name_dir_out=./results_ventri

python predict_vunet.py --outpath $name_dir_out --modelname $name_dir_model --imagepath $name_dir_in

################################################################################################

name_dir_model=./trained_models_SAH/model_total.pt  
name_dir_out=./results_total

python predict_vunet.py --outpath $name_dir_out --modelname $name_dir_model --imagepath $name_dir_in

