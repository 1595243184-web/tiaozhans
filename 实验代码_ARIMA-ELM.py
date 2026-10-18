# -*- coding: utf-8 -*-
"""ARIMA-ELM 混合预测模型 · 可复现实验脚本
================================================
复现《4 实验与验证》中的全部核心结果：
  A 纯 ARIMA          B 纯 ELM          C ARIMA+GM(1,1)
  D ARIMA+指数平滑    E ARIMA+ELM（本文）
评估方式：walk-forward 滚动预测（前15个月训练，逐月预测第16—21月）

依赖：numpy pandas statsmodels
运行：python arima_elm_experiment.py
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================ 1 数据
# 惠州市 2025-01 ~ 2026-09 月度零售均价（元/500g；食用油为元/5升桶）
# 端点、极值与关键波动节点按官方披露校准，中间月份按节点约束插值构造。
# 正式提交前请用发改局逐月原始值替换 SERIES 后重跑。
MONTHS = pd.period_range("2025-01", "2026-09", freq="M").astype(str).tolist()
SERIES = {
    "猪肉4种": [20.71, 20.63, 20.42, 20.28, 20.15, 19.92, 20.06,
              19.84, 19.60, 19.38, 19.21, 19.05,
              18.86, 18.62, 18.44, 17.46, 17.02, 15.44, 16.28, 16.05, 15.91],
    "蔬菜30种": [4.28, 4.35, 4.19, 4.10, 4.02, 4.14, 4.37,
              4.66, 4.72, 4.80, 4.91, 4.78,
              4.85, 4.72, 4.58, 4.42, 4.16, 3.98, 4.24, 4.38, 4.34],
    "鸡蛋": [6.85, 6.78, 6.72, 6.68, 6.62, 6.58, 6.55,
           6.62, 6.70, 6.75, 6.80, 6.76,
           6.70, 6.55, 6.48, 6.42, 6.40, 6.44, 6.52, 6.60, 6.73],
    "水产品25种": [31.63, 31.98, 32.40, 32.85, 33.10, 33.29, 32.76,
                32.30, 31.95, 31.70, 31.92, 32.20,
                32.55, 32.30, 31.90, 31.55, 31.30, 31.05, 31.60, 31.65, 30.48],
    "大米3种": [3.18, 3.19, 3.20, 3.19, 3.18, 3.17, 3.16,
              3.15, 3.16, 3.17, 3.18, 3.18,
              3.19, 3.18, 3.17, 3.16, 3.16, 3.17, 3.18, 3.18, 3.18],
    "食用油9种": [95.37, 95.20, 95.02, 94.88, 94.76, 94.65, 94.52,
                94.40, 94.31, 94.22, 94.10, 94.02,
                93.95, 93.88, 93.82, 93.76, 93.72, 93.70, 93.68, 93.72, 93.67],
}

TRAIN_LEN = 15     # 初始训练窗
W = 3              # 残差滞后窗
LAMBDA_GRID = [(3, 0.01), (5, 0.05), (8, 0.10), (5, 0.50)]   # (隐层节点, 正则系数)

# ============================================================ 2 工具
def mae(t, p):  return float(np.mean(np.abs(np.asarray(t) - np.asarray(p))))
def rmse(t, p): return float(np.sqrt(np.mean((np.asarray(t) - np.asarray(p)) ** 2)))
def mape(t, p): return float(np.mean(np.abs((np.asarray(t) - np.asarray(p))
                                            / np.asarray(t, float))) * 100)


class ELM:
    """极限学习机：随机输入权重 + 岭回归闭式解求输出权重。"""

    def __init__(self, n_hidden=5, lam=0.05, seed=7):
        self.L, self.lam = n_hidden, lam
        self.rng = np.random.default_rng(seed)

    def _scale(self, X, fit=False):
        if fit:
            self.xmin, self.xmax = X.min(0), X.max(0)
        rng_ = np.where((self.xmax - self.xmin) < 1e-9, 1.0, self.xmax - self.xmin)
        return (X - self.xmin) / rng_

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        Xs = self._scale(X, fit=True)
        self.ymu, self.ysd = y.mean(), y.std() + 1e-9
        d = Xs.shape[1]
        self.W_in = self.rng.normal(0, 1.0, (d, self.L))
        self.b_in = self.rng.uniform(-1, 1, self.L)
        H = np.tanh(Xs @ self.W_in + self.b_in)
        self.beta = np.linalg.solve(H.T @ H + self.lam * np.eye(self.L), H.T
                                    @ ((y - self.ymu) / self.ysd))
        return self

    def predict(self, X):
        Xs = self._scale(np.asarray(X, float))
        return np.tanh(Xs @ self.W_in + self.b_in) @ self.beta * self.ysd + self.ymu


def lag_matrix(seq, w=W):
    seq = np.asarray(seq, float)
    return np.array([seq[i - w:i] for i in range(w, len(seq))]), seq[w:]


def gm11_forecast(seq, steps=1):
    """灰色 GM(1,1) 一步预测。"""
    x0 = np.asarray(seq, float)
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack([-z1, np.ones_like(z1)])
    try:
        ab, *_ = np.linalg.lstsq(B, x0[1:].reshape(-1, 1), rcond=None)
        a, b = ab.ravel()
    except np.linalg.LinAlgError:
        return float(x0[-1])
    if abs(a) < 1e-9:
        return float(x0[-1])
    k = len(x0) + steps - 1
    f = lambda kk: (x0[0] - b / a) * np.exp(-a * kk) + b / a
    return float(f(k) - f(k - 1))


def fit_arima(train, max_pq=2):
    """在 p,q≤2、d∈{0,1} 内按 AIC 最小定阶。"""
    best = None
    for d in (1, 0):
        for p in range(max_pq + 1):
            for q in range(max_pq + 1):
                if p == 0 and q == 0 and d == 0:
                    continue
                try:
                    m = ARIMA(train, order=(p, d, q), enforce_stationarity=False,
                              enforce_invertibility=False).fit()
                except Exception:
                    continue
                if np.isfinite(m.aic) and (best is None or m.aic < best[0]):
                    best = (m.aic, (p, d, q), m)
    if best is None:
        m = ARIMA(train, order=(1, 1, 1)).fit()
        return (1, 1, 1), m
    return best[1], best[2]


def holt_forecast(seq):
    try:
        m = ExponentialSmoothing(np.asarray(seq, float), trend="add",
                                 damped_trend=True,
                                 initialization_method="estimated").fit()
        return float(np.ravel(m.forecast(1))[0])
    except Exception:
        return float(np.asarray(seq, float)[-1])


def select_elm(resid, w=W):
    """以残差末3点作时序验证，从预置网格中选 (隐层节点, 正则系数)。"""
    if len(resid) < w + 6:
        return ELM(5, 0.05)
    best, best_err = None, np.inf
    for L, lam in LAMBDA_GRID:
        Xa, ya = lag_matrix(resid[:-3], w)
        if len(ya) < 4:
            continue
        e = ELM(L, lam).fit(Xa, ya)
        Xv, yv = lag_matrix(resid, w)
        err = float(np.mean(np.abs(e.predict(Xv[-3:]) - yv[-3:])))
        if err < best_err:
            best, best_err = ELM(L, lam), err
    return best or ELM(5, 0.05)


# ============================================================ 3 滚动预测
def walk_forward(y, train_len=TRAIN_LEN):
    y = np.asarray(y, float)
    res = {k: [] for k in "ABCDE"}
    truth, orders = [], []
    for i in range(len(y) - train_len):
        tr, te = y[:train_len + i], y[train_len + i]
        truth.append(te)

        order, am = fit_arima(tr)
        orders.append(order)
        lin = float(np.ravel(am.forecast(1))[0])
        resid = tr - np.asarray(am.fittedvalues, float)
        resid = resid[np.isfinite(resid)]

        res["A"].append(lin)

        Xb, yb = lag_matrix(tr, W)
        res["B"].append(float(ELM(5, 0.05, seed=i).fit(Xb, yb)
                              .predict(tr[-W:].reshape(1, -1))[0]))

        res["C"].append(lin + gm11_forecast(resid))
        res["D"].append(lin + holt_forecast(resid))

        if len(resid) > W + 4:
            e2 = select_elm(resid)
            Xe, ye = lag_matrix(resid, W)
            e2.fit(Xe, ye)
            nl = float(e2.predict(resid[-W:].reshape(1, -1))[0])
        else:
            nl = 0.0
        res["E"].append(lin + nl)

    out = {"true": truth, "orders": orders}
    for k, v in res.items():
        out[k] = {"pred": v, "MAE": mae(truth, v), "RMSE": rmse(truth, v),
                  "MAPE": mape(truth, v), "MAXAE": float(np.max(np.abs(np.array(truth) - np.array(v))))}
    return out


# ============================================================ 4 主流程
if __name__ == "__main__":
    NAMES = {"A": "ARIMA", "B": "ELM", "C": "ARIMA+GM(1,1)",
             "D": "ARIMA+指数平滑", "E": "ARIMA+ELM(本文)"}
    detail = {}
    print("=" * 72)
    for cat, series in SERIES.items():
        r = walk_forward(series)
        detail[cat] = r
        print(f"\n【{cat}】实测第16—21月 {np.round(r['true'], 2).tolist()}")
        print(f"  ARIMA阶数 {r['orders']}")
        for k in "ABCDE":
            print(f"  {NAMES[k]:<16} MAE={r[k]['MAE']:7.4f} RMSE={r[k]['RMSE']:7.4f} "
                  f"MAPE={r[k]['MAPE']:5.2f}%  最大AE={r[k]['MAXAE']:6.3f}")

    print("\n" + "=" * 72)
    print("六品类宏平均")
    print(f"{'模型':<18}{'MAE':>9}{'RMSE':>9}{'MAPE%':>9}")
    for k in "ABCDE":
        m = np.mean([detail[c][k]["MAE"] for c in SERIES])
        r_ = np.mean([detail[c][k]["RMSE"] for c in SERIES])
        p_ = np.mean([detail[c][k]["MAPE"] for c in SERIES])
        print(f"{NAMES[k]:<18}{m:>9.4f}{r_:>9.4f}{p_:>9.2f}")

    print("\nARIMA+ELM 相对 ARIMA 的降低幅度：")
    for key in ("MAE", "RMSE", "MAPE"):
        base = np.mean([detail[c]["A"][key] for c in SERIES])
        cur = np.mean([detail[c]["E"][key] for c in SERIES])
        print(f"  {key:<5}: {base:.4f} -> {cur:.4f}   {(base - cur) / base * 100:5.2f}%")
