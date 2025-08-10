\# Brain-Based Decision Framework for WindowSurfer



\## Overview



This document describes how the new \*\*three-brain decision system\*\* works inside the WindowSurfer framework, and how it can be used or extended by other development windows and chat sessions.



The system is \*\*not\*\* a trading strategy in itself—it is a \*decision engine\* that outputs three simple, binary (True/False) signals representing the opinion of three separate "brains":



\* \*\*Bear Brain\*\*: Should the system park (stand down) entirely?

\* \*\*Chop Brain\*\*: Is this a good moment to take a mean-reversion long?

\* \*\*Bull Brain\*\*: Is this a good moment to take a momentum long?



By keeping each brain focused on \*\*one binary question\*\*, we can:



\* Measure accuracy independently for each brain.

\* Improve each brain's performance without affecting the others.

\* Combine them later for multi-strategy live trading while avoiding # Brain-Based Decision Framework for WindowSurfer



\## Overview



This document describes how the new \*\*three-brain decision system\*\* works inside the WindowSurfer framework, and how it can be used or extended by other development windows and chat sessions.



The system is \*\*not\*\* a trading strategy in itself—it is a \*decision engine\* that outputs three simple, binary (True/False) signals representing the opinion of three separate "brains":



\* \*\*Bear Brain\*\*: Should the system park (stand down) entirely?

\* \*\*Chop Brain\*\*: Is this a good moment to take a mean-reversion long?

\* \*\*Bull Brain\*\*: Is this a good moment to take a momentum long?



By keeping each brain focused on \*\*one binary question\*\*, we can:



\* Measure accuracy independently for each brain.

\* Improve each brain's performance without affecting the others.

\* Combine them later for multi-strategy live trading while avoiding conflicts.



---



\## How It Works



\### Bear Brain (PARKED?)



\*\*Goal\*\*: Detect extended downturns or high-volatility conditions where all trading should pause.



\*\*Inputs\*\*:



\* Long SMA slope (`L` = 50 default)

\* Long z-score position

\* ATR volatility cooling



\*\*Fire Rule\*\*:



\* Return `True` (park) if:



&nbsp; \* Slope of SMA\\\_L < 0 \*\*or\*\*

&nbsp; \* z\\\_L < -1.0

\* Resume only if:



&nbsp; \* Slope >= 0 \*\*and\*\* |z\\\_L| <= 0.5 \*\*and\*\* ATR has cooled for `dwell` candles.



---



\### Chop Brain (EDGE\\\_LONG?)



\*\*Goal\*\*: Harvest 4–8% swings inside sideways/choppy markets.



\*\*Inputs\*\*:



\* Short-window z-score (`S` = 20 default)

\* Local SMA slope (`slope\_w` = 5 default)



\*\*Fire Rule\*\*:



\* Return `True` (long) if:



&nbsp; \* z\\\_S <= -1.0 \*\*and\*\*

&nbsp; \* Slope(SMA\\\_5) >= 0



\*\*Exit Condition (for trading contexts)\*\*:



\* Take profit at +4% min; cap at +8%

\* Sell earlier if z\\\_S >= +1.0 and slope <= 0



---



\### Bull Brain (MOMO\\\_LONG?)



\*\*Goal\*\*: Catch continuation in strong upward trends.



\*\*Inputs\*\*:



\* Medium SMA slope (`M` = 20 default)

\* Higher-highs count in recent window (`hh\_lookback` = 5 default)



\*\*Fire Rule\*\*:



\* Return `True` (long) if:



&nbsp; \* slope(SMA\\\_M) > 0 \*\*and\*\*

&nbsp; \* close > SMA\\\_M \*\*and\*\*

&nbsp; \* Higher-high count >= 3 in last 5 candles



\*\*Exit Condition (for trading contexts)\*\*:



\* Take profit at +6%; stop at -4%



---



\## Using in Other Windows \& Chat



Because each brain outputs a \*\*boolean decision\*\* given price series context, they can be plugged into:



### Quick Brain Check (single coin, full history, no files)

```powershell
python bot.py brains score --coin SOLUSDT --no_write -vvv
```

--coin SOLUSDT → test one coin  
--no_write → print-only (no CSVs)  
-vvv → step-by-step debug; use -vv for periodic progress  
Press Ctrl+C anytime — you’ll get a summary of Bear/Chop/Bull hit rates so far.

\* Simulation harnesses to measure historical hit-rate.

\* Live bots to gate trade execution.

\* Regime detectors to switch between strategies.

\* Performance dashboards for drift monitoring.



\*\*Example integration\*\*:



```python

from systems.brains import bear, chop, bull



\# Example per-candle loop

t = 500  # candle index

if bear.parked(t, series, cfg):

&nbsp;   # Skip all trading

&nbsp;   pass

elif chop.edge\_long(t, series, cfg):

&nbsp;   # Signal mean-reversion long

&nbsp;   pass

elif bull.momo\_long(t, series, cfg):

&nbsp;   # Signal momentum long

&nbsp;   pass

```



---



\## Testing and Measuring Success



The accuracy of each brain is tracked separately:



1\. Run the brain on historical data for multiple coins.

2\. When it returns `True`, check if the defined future condition was met.

3\. Compute:



&nbsp;  \* Hit rate ($\\hat{p}$)

&nbsp;  \* Wilson 95% CI

&nbsp;  \* Lift over baseline (naïve up/down rate)

4\. Monitor rolling hit rate for drift.



A brain is considered \*\*healthy\*\* if it maintains ≥ 70% hit rate across recent slices.



---



\## Key Principles



\* \*\*One Brain, One Job\*\*: Each brain makes exactly one type of decision.

\* \*\*Event-Based Evaluation\*\*: Only score decisions when the brain actually fires.

\* \*\*Regime Independence\*\*: Brains can be combined or run in isolation.

\* \*\*Simplicity First\*\*: Rules are easy to interpret and tune.



---



\## Future Extensions



\* \*\*Bear Brain\*\* as a global risk filter for any trading module.

\* \*\*Chop Brain\*\* paired with a volatility filter to avoid entering pre-breakout conditions.

\* \*\*Bull Brain\*\* integrated with regime classification for adaptive thresholds.

\* Drift detection alerts to re-tune brains when accuracy degrades.



---



With this setup, WindowSurfer's brains can be tested, refined, and deployed independently, ensuring that live trading decisions are grounded in proven, measurable signals.

conflicts.



---



\## How It Works



\### Bear Brain (PARKED?)



\*\*Goal\*\*: Detect extended downturns or high-volatility conditions where all trading should pause.



\*\*Inputs\*\*:



\* Long SMA slope (`L` = 50 default)

\* Long z-score position

\* ATR volatility cooling



\*\*Fire Rule\*\*:



\* Return `True` (park) if:



&nbsp; \* Slope of SMA\\\_L < 0 \*\*or\*\*

&nbsp; \* z\\\_L < -1.0

\* Resume only if:



&nbsp; \* Slope >= 0 \*\*and\*\* |z\\\_L| <= 0.5 \*\*and\*\* ATR has cooled for `dwell` candles.



---



\### Chop Brain (EDGE\\\_LONG?)



\*\*Goal\*\*: Harvest 4–8% swings inside sideways/choppy markets.



\*\*Inputs\*\*:



\* Short-window z-score (`S` = 20 default)

\* Local SMA slope (`slope\_w` = 5 default)



\*\*Fire Rule\*\*:



\* Return `True` (long) if:



&nbsp; \* z\\\_S <= -1.0 \*\*and\*\*

&nbsp; \* Slope(SMA\\\_5) >= 0



\*\*Exit Condition (for trading contexts)\*\*:



\* Take profit at +4% min; cap at +8%

\* Sell earlier if z\\\_S >= +1.0 and slope <= 0



---



\### Bull Brain (MOMO\\\_LONG?)



\*\*Goal\*\*: Catch continuation in strong upward trends.



\*\*Inputs\*\*:



\* Medium SMA slope (`M` = 20 default)

\* Higher-highs count in recent window (`hh\_lookback` = 5 default)



\*\*Fire Rule\*\*:



\* Return `True` (long) if:



&nbsp; \* slope(SMA\\\_M) > 0 \*\*and\*\*

&nbsp; \* close > SMA\\\_M \*\*and\*\*

&nbsp; \* Higher-high count >= 3 in last 5 candles



\*\*Exit Condition (for trading contexts)\*\*:



\* Take profit at +6%; stop at -4%



---



\## Using in Other Windows \& Chat



Because each brain outputs a \*\*boolean decision\*\* given price series context, they can be plugged into:



\* Simulation harnesses to measure historical hit-rate.

\* Live bots to gate trade execution.

\* Regime detectors to switch between strategies.

\* Performance dashboards for drift monitoring.



\*\*Example integration\*\*:



```python

from systems.brains import bear, chop, bull



\# Example per-candle loop

t = 500  # candle index

if bear.parked(t, series, cfg):

&nbsp;   # Skip all trading

&nbsp;   pass

elif chop.edge\_long(t, series, cfg):

&nbsp;   # Signal mean-reversion long

&nbsp;   pass

elif bull.momo\_long(t, series, cfg):

&nbsp;   # Signal momentum long

&nbsp;   pass

```



---



\## Testing and Measuring Success



The accuracy of each brain is tracked separately:



1\. Run the brain on historical data for multiple coins.

2\. When it returns `True`, check if the defined future condition was met.

3\. Compute:



&nbsp;  \* Hit rate ($\\hat{p}$)

&nbsp;  \* Wilson 95% CI

&nbsp;  \* Lift over baseline (naïve up/down rate)

4\. Monitor rolling hit rate for drift.



A brain is considered \*\*healthy\*\* if it maintains ≥ 70% hit rate across recent slices.



---



\## Key Principles



\* \*\*One Brain, One Job\*\*: Each brain makes exactly one type of decision.

\* \*\*Event-Based Evaluation\*\*: Only score decisions when the brain actually fires.

\* \*\*Regime Independence\*\*: Brains can be combined or run in isolation.

\* \*\*Simplicity First\*\*: Rules are easy to interpret and tune.



---



\## Future Extensions



\* \*\*Bear Brain\*\* as a global risk filter for any trading module.

\* \*\*Chop Brain\*\* paired with a volatility filter to avoid entering pre-breakout conditions.

\* \*\*Bull Brain\*\* integrated with regime classification for adaptive thresholds.

\* Drift detection alerts to re-tune brains when accuracy degrades.



---



With this setup, WindowSurfer's brains can be tested, refined, and deployed independently, ensuring that live trading decisions are grounded in proven, measurable signals.



