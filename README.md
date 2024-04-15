# CSP-MvCDA
% Python code for Class-Structure Preserving Multi-View Correlated Discriminant Analysis for Multiblock Data (CSP-MvCDA)          
% Author: Sankar Mondal and Pradipta maji                                               
% Email: sankar_r@isical.ac.in (Sankar Mondal)                                                     
% Date: February, 2024                                                                                                                               


There are three folders, namely, Code, Data and Result
Code folder contains the code files for CSP-MvCDA
Data folder contains one example data set ALOI (https://elki-project.github.io/datasets/multi_view)
Result folder contain the result correspoding to ALOI

To run the code make sure you use the suitable parameters. Here, a list is provided
alpha_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
beta_list = [0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for example, alpha = 0.4 and beta = 0.001 is taken for ALOI
The best result is obtained by exhaustively searching the parameter space.

Example:
Move into the code folder and use the following to execute the code CSP-MvCDA.py on the data set ALOI

user_name$: python3 CSP_MvCDA.py 0.001 0.4

## Make sure the data takes place in the data directory and follow the same format as directed.
