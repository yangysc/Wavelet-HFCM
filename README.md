# Purpose
This project is the source code for the paper *Time Series Forecasting based on High-Order Fuzzy Cognitive Maps and Wavelet Transform*, which is now published in **[IEEE TFS](https://ieeexplore.ieee.org/document/8352858)**.


# Usage
- The file ***Wavelet_HFCM.py*** is the main program to perform forecasting time series by using Wavelet-HFCM.
- Defining the basic functions of an FCM, *FCMs.py* is used in the main program, and there is no need to run it seperately.
- The outcomes about **the effects of two hyper-parameters** (the order *k* and number of nodes *Nc*）on Wavelet-HFCM are saved into the file *Outcome_for_papers/output_sunspot_sp500.xlsx*, and their corresponding plots are saved into the directory *./Outcome_for_papers/impact_parameters*/  .


# Requirements
- python 3.+
- matplotlib
- pandas
- numpy
