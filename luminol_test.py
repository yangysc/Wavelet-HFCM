from luminol.anomaly_detector import AnomalyDetector
import pandas as pd
import numpy as np

src = '/home/shanchao/Documents/For_research/By_Python/HFCM/NAB-master/data/realTraffic/occupancy_6005.csv'
ts = pd.read_csv(src, delimiter=',').as_matrix()[:, 1]
ts = np.array(ts, dtype=np.float)
time = range(len(ts))
my_detector = AnomalyDetector(ts)
score = my_detector.get_all_scores()
