# Purpose
This project is the source code for the paper *Time Series Forecasting based on High-Order Fuzzy Cognitive Maps and Wavelet Transform*, which is now published in **[IEEE TFS](https://ieeexplore.ieee.org/document/8352858)**.


# Usage
- The file ***Wavelet_HFCM.py*** is the main program to perform forecasting time series by using Wavelet-HFCM.
- Defining the basic functions of an FCM, *FCMs.py* is used in the main program, and there is no need to run it seperately.
- The outcomes about **the effects of two hyper-parameters** (the order *k* and number of nodes *Nc*）on Wavelet-HFCM are saved into the file *Outcome_for_papers/output_sunspot_sp500.xlsx*, and their corresponding plots are saved into the directory *./Outcome_for_papers/impact_parameters*/  .


# Requirements
- python (3.6)
- matplotlib (3.0.3)
- seaborn (0.9.0)
- pandas (0.24.2)
- numpy (1.16.3)



# Example

Here is an example for MG-chaos data.
![image](https://github.com/yangysc/Wavelet-HFCM/blob/master/Outcome_for_papers/impact_parameters/MG-chaos.png)

# Citation
If you find this work useful, please cite our paper:
```
@article{yang2018time,
  title={Time-series forecasting based on high-order fuzzy cognitive maps and wavelet transform},
  author={Yang, Shanchao and Liu, Jing},
  journal={IEEE Transactions on Fuzzy Systems},
  volume={26},
  number={6},
  pages={3391--3402},
  year={2018},
  publisher={IEEE}
}
```


