# DiTauDetection
README FOR DI-TAU PARTICLE DETECTION PROJECT

General overview:
In this project we implemented deepset networks, simple cnn and tested common variational encoders for image
classification in latent domain using support vector machines. Moreover, we implemented infrastructure and began testing
the auto encoders in anomaly detection and implemented focal loss and some supportive functions for Ridnik asymmetric
loss. Also we created data generator that normalizes and reforms the data from the csv into RGB images, row vectors for
deepset analysis and future possible BERT analysis/classification.

Files:

DitauDataGenerator.py  - this file parsing the csv file, extracts the normalization info and then packs the data into
pkl files that contain the label, track, em and had vectors, 3*64*64 images that represents the sample.

DitauDataLoader.py - this file has the data loaders and the classes that represents the datasets. we save multiple files
one for each sample to overcome RAM limitations.

DataVisualization.py - this file has most of functions to plot the graps and graphicall y compare our results.

DeepSetExtended.py - this file has the deepset architecture that matched the data generated based on the ATLAS sensor
with merged data for the Had and EM sensors. In the new generated data the Had and EM are not necessarily at the same
coordinates.

DitauCNN - Simple CNN for the image classification to run in parallel to the trancks to give some additional accuracy.
Due to physics consideration the data in comming from the EM and Had and mapped to the image is significantly less
valuable, so we didn't implement big and heavy NN for this.

DiTauVAE.py - This file has all the routines to train the VAE, the SVMs, evaluate the performance for classic training
and for anomaly detection.

FocalLoss.py - implemented the focal loss
ridnik_asymetric_loss.py - implementation of the ridnik ASL with minor modifications to fit our data. We implemented the
adaptive losses from scratch in DeepLearningProject.py

TripleDeepset.py - the implemnentation of the network that runs three parallel deepset networks on tracks, EM and Had
and returns a single decision.

VAE_models - folder with the common implemented variational auto encoders.

Main.py - this file runs the data generation, configurations, networks training and evaluations, comparisons,
VAE training and evaluation + anomaly detection.

* To run the different modes you need to change the MODE at the beginning of the Main.py file.
* The file is structured in a way that it may be easily transformed to jupiter notebook or to be separated to functions
* We attached some already pre-trained networks in NetworkParameters.
* csv - https://cernbox.cern.ch/index.php/s/y4xhcdH4Xv4SK6n
