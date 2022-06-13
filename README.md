# MADE
MADE (Mass, Age, and Distance Estimator) estimates the mass, age, metallicity, and distance of stars given the labels: H-band apparent magnitude,J-K colour, parallax, log g, Teff, M/H, a/M, C/M, N/M. The saved neural network has been trained on APOKASC (APOGEE DR14) and TGAS data. The network can easily retrained with new data.

CalcBNN3Layer.py - This module uses PyMC3 to train a Bayesian neural network with a single hidden layer, and then applies it to make predictions in the case of unknown and known targets.

TrainApokascTgasDR14AgeDistBNN.ipynb - Notebook with particular application that trains a neural network with a single hidden layer to APOKASC (APOGEE DR14) and TGAS data.

CalcApogeeTgasDR14AgeDistBNN.ipynb - Notebook applying Bayesian neural network on APOGEE DR14-TGAS data to estimate masses, ages, metallicities, and distances.
