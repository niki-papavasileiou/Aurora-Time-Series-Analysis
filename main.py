import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv("latest_HPI.txt",delim_whitespace=True)                           

df['Observation'] = pd.to_datetime(df['Observation'], format='%Y-%m-%d_%H:%M')

df['Forecast'] = pd.to_datetime(df['Forecast'], format='%Y-%m-%d_%H:%M')

df.index = df['Forecast']
#del df['Forecast']

df.index = df['Observation']

ax = df.plot(x="Observation", y="North-Hemispheric-Power-Index", legend=False)
ax2 = ax.twinx()
df.plot(x="Observation", y="South-Hemispheric-Power-Index-GigaWatts", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()

autocorrelation_lag1 = df['North-Hemispheric-Power-Index'].autocorr(lag=12)
print("North Hemispheric Power Index one hour Lag: ", autocorrelation_lag1)

autocorrelation_lag2 = df['South-Hemispheric-Power-Index-GigaWatts'].autocorr(lag=12)
print("South Hemispheric Power Index one hour Lag: ", autocorrelation_lag2)

autocorrelation_lag3 = df['North-Hemispheric-Power-Index'].autocorr(lag=1)
print("North Hemispheric Power Index 5 min Lag: ", autocorrelation_lag3)

autocorrelation_lag4 = df['South-Hemispheric-Power-Index-GigaWatts'].autocorr(lag=1)
print("South Hemispheric Power Index 5 min Lag: ", autocorrelation_lag4)

decompose = seasonal_decompose(df['North-Hemispheric-Power-Index'],model='additive', period=12)
decompose.plot()
plt.show()

decompose = seasonal_decompose(df['South-Hemispheric-Power-Index-GigaWatts'],model='additive', period=12)
decompose.plot()
plt.show()