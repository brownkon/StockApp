"""
Backtesting Engine
-------------------
Uses Backtrader to simulate the ML-driven trading strategy
on out-of-sample data and generate performance reports.
"""

import argparse
import csv
import logging
import os
from datetime import datetime

import backtrader as bt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# --------------------------------------------------------------------------- #
#  Custom Backtrader Data Feed (from our feature parquet)
# --------------------------------------------------------------------------- #
class ParquetData(bt.feeds.PandasData):
    """Backtrader data feed from our feature parquet files."""
    params = (
        ("datetime", None),   # Use the index
        ("open", "close"),    # We only have close in features; approximate
        ("high", "close"),
        ("low", "close"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


# --------------------------------------------------------------------------- #
#  ML Signal Strategy
# --------------------------------------------------------------------------- #
class MLSignalStrategy(bt.Strategy):
    """Strategy that trades based on pre-computed ML predictions.
    
    Reads the predictions DataFrame and executes trades when
    the model predicts a positive 5-day return with sufficient
    confidence.
    """

    params = (
        ("predictions", None),        # DataFrame with predictions per ticker
        ("prob_threshold", 0.55),      # Minimum probability to enter
        ("max_positions", 5),          # Max concurrent positions
        ("atr_multiplier", 2.0),       # Trailing stop = atr_multiplier * ATR
        ("ticker_map", None),          # Map from ticker name to data feed index
    )

    def __init__(self):
        self.order_dict = {}
        self.trade_log = []
        self.entry_prices = {}
        self.entry_dates = {}

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.debug(f"{dt} - {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        ticker = order.data._name
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"BUY  {ticker} @ {order.executed.price:.2f}")
                self.entry_prices[ticker] = order.executed.price
                self.entry_dates[ticker] = self.datas[0].datetime.date(0)
            elif order.issell():
                self.log(f"SELL {ticker} @ {order.executed.price:.2f}")

                # Log the trade
                entry_price = self.entry_prices.pop(ticker, order.executed.price)
                entry_date = self.entry_dates.pop(ticker, self.datas[0].datetime.date(0))
                exit_price = order.executed.price
                exit_date = self.datas[0].datetime.date(0)
                pnl_pct = (exit_price - entry_price) / entry_price * 100

                self.trade_log.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date),
                    "exit_date": str(exit_date),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "duration_days": (exit_date - entry_date).days,
                })

        self.order_dict.pop(ticker, None)

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        preds = self.p.predictions

        if preds is None or preds.empty:
            return

        # Count current open positions
        open_positions = sum(
            1 for d in self.datas if self.getposition(d).size > 0
        )

        for i, data in enumerate(self.datas):
            ticker = data._name
            pos = self.getposition(data)

            # Get prediction for this ticker and date
            pred_row = preds[
                (preds["ticker"] == ticker) &
                (preds["date"] == pd.Timestamp(current_date))
            ]

            if pred_row.empty:
                # No prediction for this date — if we have a position, hold it
                continue

            signal = pred_row.iloc[0]["predicted_signal"]
            prob = pred_row.iloc[0]["predicted_probability"]

            # Skip if we have a pending order for this ticker
            if ticker in self.order_dict:
                continue

            if pos.size == 0:
                # Not in position — check for entry
                if (
                    signal == 1
                    and prob >= self.p.prob_threshold
                    and open_positions < self.p.max_positions
                ):
                    # Equal-weight sizing
                    cash = self.broker.getcash()
                    size_value = cash / self.p.max_positions
                    price = data.close[0]
                    if price > 0:
                        size = int(size_value / price)
                        if size > 0:
                            order = self.buy(data=data, size=size)
                            self.order_dict[ticker] = order
                            open_positions += 1
            else:
                # In position — check for exit
                if signal == 0 or prob < self.p.prob_threshold:
                    order = self.sell(data=data, size=pos.size)
                    self.order_dict[ticker] = order


# --------------------------------------------------------------------------- #
#  Backtest Runner
# --------------------------------------------------------------------------- #
def load_predictions_for_backtest(test_start="2024-01-01", test_end=None):
    """Load all feature data and generate predictions for the test period.
    
    For backtesting, we use the actual model predictions that would have been
    available at each point in time. We simulate this by using the saved
    signal column from our features (which is the ground truth), but for
    proper backtesting, we should retrain or use walk-forward predictions.
    
    For now, we load the model and generate predictions across the test period.
    """
    import joblib

    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    model = joblib.load(model_path)

    # Load all features
    all_path = os.path.join(DATA_DIR, "all_features.parquet")
    if not os.path.exists(all_path):
        raise FileNotFoundError(
            "Feature data not found. Run build_features.py first."
        )

    df = pd.read_parquet(all_path)
    df.index = pd.to_datetime(df.index)

    # Filter to test period
    start = pd.Timestamp(test_start)
    if test_end:
        end = pd.Timestamp(test_end)
        test_df = df[(df.index >= start) & (df.index <= end)]
    else:
        test_df = df[df.index >= start]

    if test_df.empty:
        raise ValueError("No data in the test period")

    # Get feature columns from the model to ensure correct ordering
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        target_cols = {"fwd_return_1d", "fwd_return_5d", "fwd_return_10d", "signal",
                       "signal_1d", "signal_5d", "signal_10d"}
        meta_cols = {"ticker"}
        feature_cols = [c for c in test_df.columns if c not in target_cols and c not in meta_cols]

    # Ensure all expected columns exist, fill missing with NaN
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = float("nan")

    # Generate predictions
    X = test_df[feature_cols]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    pred_df = pd.DataFrame({
        "ticker": test_df["ticker"].values,
        "date": test_df.index,
        "predicted_signal": preds,
        "predicted_probability": probs,
    })

    logger.info(
        f"Generated {len(pred_df)} predictions for test period "
        f"({test_start} to {test_end or 'present'})"
    )
    return pred_df, test_df


def run_backtest(
    test_start="2024-01-01",
    test_end=None,
    initial_cash=10000.0,
    prob_threshold=0.55,
    max_positions=5,
    commission=0.0,
    slippage_pct=0.0001,
):
    """Execute the full backtest and generate reports."""

    pred_df, test_df = load_predictions_for_backtest(test_start, test_end)

    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(
        MLSignalStrategy,
        predictions=pred_df,
        prob_threshold=prob_threshold,
        max_positions=max_positions,
    )

    # Add data feeds for each ticker
    tickers = sorted(test_df["ticker"].unique())
    for ticker in tickers:
        ticker_df = test_df[test_df["ticker"] == ticker][["close", "volume"]].copy()
        if ticker_df.empty:
            continue

        ticker_df.index = pd.to_datetime(ticker_df.index)
        ticker_df = ticker_df.sort_index()

        data = ParquetData(dataname=ticker_df, name=ticker)
        cerebro.adddata(data)

    # Broker settings
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    if slippage_pct > 0:
        cerebro.broker.set_slippage_perc(slippage_pct)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.04/252)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    # Run
    logger.info(f"Starting backtest with ${initial_cash:,.0f} capital...")
    results = cerebro.run()
    strategy = results[0]

    # Extract metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trades_analysis = strategy.analyzers.trades.get_analysis()
    returns_analysis = strategy.analyzers.returns.get_analysis()

    sharpe_ratio = sharpe.get("sharperatio", None)
    max_dd = drawdown.get("max", {}).get("drawdown", 0)

    total_trades = trades_analysis.get("total", {}).get("total", 0)
    won_trades = trades_analysis.get("won", {}).get("total", 0)
    lost_trades = trades_analysis.get("lost", {}).get("total", 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    gross_profit = trades_analysis.get("won", {}).get("pnl", {}).get("total", 0)
    gross_loss = abs(trades_analysis.get("lost", {}).get("pnl", {}).get("total", 0.01))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Compute benchmark (SPY buy-and-hold)
    spy_return = _compute_benchmark_return(test_df, test_start, test_end)

    # Annualized return
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end) if test_end else test_df.index.max()
    days = (test_end_dt - test_start_dt).days
    years = days / 365.25
    annualized = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100

    # Average trade duration
    trade_log = strategy.trade_log
    avg_duration = 0
    if trade_log:
        durations = [t["duration_days"] for t in trade_log]
        avg_duration = sum(durations) / len(durations)

    metrics = {
        "initial_capital": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized, 2),
        "spy_benchmark_return_pct": round(spy_return, 2),
        "sharpe_ratio": round(sharpe_ratio, 4) if sharpe_ratio else None,
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "∞",
        "avg_trade_duration_days": round(avg_duration, 1),
        "test_period": f"{test_start} to {test_end or 'present'}",
        "prob_threshold": prob_threshold,
        "max_positions": max_positions,
    }

    # Generate outputs
    _save_trade_log(trade_log)
    _generate_report(metrics, trade_log)
    _save_equity_curve(cerebro, strategy)

    # Print summary
    _print_summary(metrics)

    return metrics, trade_log


def _compute_benchmark_return(test_df, test_start, test_end):
    """Compute buy-and-hold SPY return over the test period."""
    spy_df = test_df[test_df["ticker"] == "SPY"].copy()
    if spy_df.empty:
        logger.warning("No SPY data for benchmark")
        return 0.0

    spy_df = spy_df.sort_index()
    start_price = spy_df["close"].iloc[0]
    end_price = spy_df["close"].iloc[-1]
    return (end_price - start_price) / start_price * 100


def _save_trade_log(trade_log):
    """Save trade log to CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "backtest_trades.csv")

    if not trade_log:
        logger.info("No trades to log")
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade_log[0].keys())
        writer.writeheader()
        writer.writerows(trade_log)

    logger.info(f"Trade log saved to {csv_path}")


def _generate_report(metrics, trade_log):
    """Generate markdown performance report."""
    os.makedirs(DATA_DIR, exist_ok=True)
    report_path = os.path.join(DATA_DIR, "backtest_report.md")

    lines = [
        "# Backtest Performance Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Strategy Configuration",
        f"- **Test Period:** {metrics['test_period']}",
        f"- **Initial Capital:** ${metrics['initial_capital']:,.0f}",
        f"- **Probability Threshold:** {metrics['prob_threshold']}",
        f"- **Max Positions:** {metrics['max_positions']}",
        "",
        "## Performance Summary",
        "",
        "| Metric | Value |",
        "|:-------|------:|",
        f"| Final Value | ${metrics['final_value']:,.2f} |",
        f"| Total Return | {metrics['total_return_pct']:.2f}% |",
        f"| Annualized Return | {metrics['annualized_return_pct']:.2f}% |",
        f"| SPY Benchmark | {metrics['spy_benchmark_return_pct']:.2f}% |",
        f"| Sharpe Ratio | {metrics['sharpe_ratio'] or 'N/A'} |",
        f"| Max Drawdown | {metrics['max_drawdown_pct']:.2f}% |",
        f"| Total Trades | {metrics['total_trades']} |",
        f"| Win Rate | {metrics['win_rate_pct']:.2f}% |",
        f"| Profit Factor | {metrics['profit_factor']} |",
        f"| Avg Trade Duration | {metrics['avg_trade_duration_days']:.1f} days |",
        "",
    ]

    if trade_log:
        lines.extend([
            "## Trade Log (Last 20 Trades)",
            "",
            "| Ticker | Entry Date | Exit Date | Entry $ | Exit $ | P&L % | Days |",
            "|:-------|:-----------|:----------|--------:|-------:|------:|-----:|",
        ])
        for t in trade_log[-20:]:
            lines.append(
                f"| {t['ticker']} | {t['entry_date']} | {t['exit_date']} | "
                f"{t['entry_price']:.2f} | {t['exit_price']:.2f} | "
                f"{t['pnl_pct']:.2f}% | {t['duration_days']} |"
            )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {report_path}")


def _save_equity_curve(cerebro, strategy):
    """Save equity curve plot."""
    os.makedirs(DATA_DIR, exist_ok=True)
    plot_path = os.path.join(DATA_DIR, "backtest_equity_curve.png")

    try:
        fig = cerebro.plot(style="candlestick", iplot=False)[0][0]
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Equity curve saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not save equity curve plot: {e}")


def _print_summary(metrics):
    """Print a formatted summary to console."""
    print("\n" + "=" * 60)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Period:              {metrics['test_period']}")
    print(f"  Initial Capital:     ${metrics['initial_capital']:>12,.0f}")
    print(f"  Final Value:         ${metrics['final_value']:>12,.2f}")
    print("-" * 60)
    print(f"  Total Return:        {metrics['total_return_pct']:>11.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return_pct']:>11.2f}%")
    print(f"  SPY Benchmark:       {metrics['spy_benchmark_return_pct']:>11.2f}%")
    print(f"  Sharpe Ratio:        {str(metrics['sharpe_ratio'] or 'N/A'):>12}")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>11.2f}%")
    print("-" * 60)
    print(f"  Total Trades:        {metrics['total_trades']:>12}")
    print(f"  Win Rate:            {metrics['win_rate_pct']:>11.2f}%")
    print(f"  Profit Factor:       {str(metrics['profit_factor']):>12}")
    print(f"  Avg Duration:        {metrics['avg_trade_duration_days']:>9.1f} days")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading strategy backtest")
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Backtest start date (default: 2024-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Backtest end date (default: latest available)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Min probability threshold to enter a trade (default: 0.55)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Max concurrent positions (default: 5)",
    )
    args = parser.parse_args()
    run_backtest(
        test_start=args.start,
        test_end=args.end,
        initial_cash=args.cash,
        prob_threshold=args.threshold,
        max_positions=args.max_positions,
    )
