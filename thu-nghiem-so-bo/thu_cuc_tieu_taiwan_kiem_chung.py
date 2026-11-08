# -*- coding: utf-8 -*-
"""Chẩn đoán số cực tiểu J_m (m = 2, 30 khởi tạo) trên Taiwan Default: mẫu kiểm chứng (log / chưa log) và mẫu thiết kế
(log), c = 3, 5, 6, 8, 10. Cùng cách lấy mẫu với thu_8_cach_ket_hop.py.

Chạy: python3 thu_cuc_tieu_taiwan_kiem_chung.py
"""
import os, sys, glob
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from thu_pso_sau import FCM, RAW, signed_log, zscore
t = pd.read_excel(glob.glob(RAW + "/taiwan-default/*.xls")[0], header=1)
y = t.iloc[:, -1].values; X = t.drop(columns=[t.columns[0], t.columns[-1]]).values.astype(float)
rng = np.random.default_rng(0)
cu = np.concatenate([rng.choice(np.where(y == k)[0], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
con_lai = np.setdiff1d(np.arange(len(y)), cu); rng = np.random.default_rng(1)
moi = np.concatenate([rng.choice(con_lai[y[con_lai] == k], int(round(5000 * np.mean(y == k))), replace=False) for k in (0, 1)])
D = {"moi_log": zscore(signed_log(X[moi])), "moi_khong_log": zscore(X[moi]), "cu_log": zscore(signed_log(X[cu]))}
def run(a):
    n, c = a; Xd = D[n]; J=[]
    for s in range(30):
        mdl = FCM(c, m=2.0, max_iter=3000); _, V, _ = mdl.fit(Xd, seed=s); J.append(mdl.objective_from_centroids(Xd, V))
    J = np.sort(J); mins=[J[0]]
    for j in J[1:]:
        if j - mins[-1] > 1e-4*abs(mins[-1]): mins.append(j)
    return n, c, len(mins), float(np.mean(J <= J[0]*(1+1e-4))), (J[-1]-J[0])/J[0]
if __name__ == "__main__":
    with Pool(8) as p:
        for r in p.map(run, [(n,c) for n in D for c in (3,5,6,8,10)]):
            print("%-14s c=%-2d số cực tiểu=%d  tỉ lệ đạt J tốt nhất=%.2f  chênh J xấu nhất=%.4f" % r)
