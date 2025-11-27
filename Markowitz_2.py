"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0, ma_window = 200):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma
        self.ma_window = ma_window

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        # 確保 Gurobi 環境每次都是乾淨的
        with gp.Env(empty=True) as env:
            # 關閉 Gurobi 輸出，避免洗版
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                
                # 1. 決策變數 w (權重向量)，lb=0.0 為做多限制 (Long-only)
                w = model.addMVar(n, name="w", lb=0.0)
                
                # 2. 目標函數: Maximize (Expected Return - Risk Penalty)
                # Maximize (mu^T @ w - (gamma / 2) * (w^T @ Sigma @ w))
                model.setObjective(mu @ w - (gamma / 2) * (w @ Sigma @ w), gp.GRB.MAXIMIZE)

                # 3. 限制條件: 權重總和必須為 1
                model.addConstr(w.sum() == 1, name="Budget")
                
                model.optimize()

                # 4. 提取最佳解
                if model.status == gp.GRB.OPTIMAL:
                    # 成功求解，回傳權重
                    return w.X.tolist()
                else:
                    # 求解失敗或無法確定，回傳等權重作為備案
                    return [1/n] * n

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
       # 計算 SPY 的 200 天移動平均線 (MA)
        spy_ma = self.price[self.exclude].rolling(window=self.ma_window).mean()
        
        for i in range(self.lookback, len(df)): 
            current_date = df.index[i]
            
            # C. 取得市場訊號
            current_spy_price = self.price[self.exclude].loc[current_date]
            current_spy_ma = spy_ma.loc[current_date]
            
            # D. 動態決定 Gamma 值
            if current_spy_price > current_spy_ma:
                # 強勢市場：積極追求報酬，較低的風險厭惡 (低 gamma)
                self.gamma = 0.0
            else:
                # 弱勢市場/避險：積極避險，極高的風險厭惡 (高 gamma)
                self.gamma = 5.0 # 此值可微調，越大越趨近最小變異數組合
            
            # E. 準備給 Gurobi 的資料
            # 取得過去 lookback 天的報酬率 (不含當天 i)
            R_n = self.returns.copy()[assets].iloc[i - self.lookback : i]

            # F. 呼叫 MV 最佳化求解器
            weights = self.mv_opt(R_n, self.gamma) 
            
            # G. 填入權重
            self.portfolio_weights.loc[current_date, assets] = weights
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
