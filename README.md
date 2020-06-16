
DISCO Cell Localization CNN Scripts
=
This repository contains the relevant files for generation and visualization of the convolutional neural network (CNN) presented in the forthcoming publication indicated at the bottom of this page. 

Scripts
-
* kfold_hyperopt_binary_localizer.py - Uses the hyperopt package to explore and optimize the hyperparameter space of the CNN. 
* optimized_binary_localizer.py - Trains the final network with the optimized (and manually selected) hyperparameters. 

Notebooks
-
* genetic_optimization_visualizer.ipynb - Shows the results of the 'kfold_hyperopt_binary_localizer.py' script for selection of the optimal network hyperparameters. 
* training_log_plotter.ipynb - Plots the training history of the optimized network. 
* output_visualizer.ipynb - Loads the final network and shows segmentation predictions on validation data as well as the necessary downstream processing for laser vector path generation. 

Data
-
* binary_data.p -  The cell image data and annotations used as input to the network (training and validation).
* genetic_trials_cv.p - Output from 'kfold_hyperopt_binary_localizer.py' showing the optimization trials.
* binary_localizer_16_0.28892_1_54_7_12.log - The training log produced by 'optimized_binary_localizer.py'
* binary_localizer_16_0.28892_1_54_7_12.hdf5 - The final network used for automated cell laser lysis. 


**Note** these scripts were tested on:
* tensorflow==1.13.1
* hyperopt==0.2.3
* keras==2.2.4
* numpy==1.16.2
* opencv-python==3.3.0
* sklearn==0.20.1

Publication:
=
Lamanna, J.* , Scott, E.Y. * , Edwards, H.* , Chamberlain, M.D., Dryden, M.D.M., Peng, J., Mair, B., Lee, A., Sklavounos, A.A., Abbas, F., Moffat, J. & A.R. Wheeler. "Digital Microfluidic Isolation of Single Cells for - Omics". 2020.  
\* Co-first authors
DOI: (TBD)

