import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

outdata = "figs/test.pdf"
indata = "out/data-1592872973.421909.csv"
title = "Routing Problem"
UE = None
SO = None

df = pd.read_csv(indata)
cols = df.columns
print(df.head())


plt.style.use('ggplot')
plt.figure()
plt.title(title)

if UE is not None:
    plt.axhline(UE, color='red', linestyle='dotted', label='User Equilibrium', linewidth=2)
if SO is not None:
    plt.axhline(SO, color='black', linestyle='dotted', label='System Optimum', linewidth=2)

if 'ql_mean' in df:
    ql_mean = df['ql_mean'].values
    episodes = len(ql_mean)
    plt.plot(np.arange(episodes), ql_mean, label='QL')
if 'ga_mean' in df:
    ga_mean = df['ga_mean'].values
    episodes = len(ga_mean)
    plt.plot(np.arange(episodes), ga_mean, label='GA')

coop_ql_mean = df['coop_ql_mean'].values
coop_ga_mean = df['coop_ga_mean'].values
episodes = len(coop_ga_mean)
plt.plot(np.arange(episodes), coop_ql_mean, label='QL under GA<->QL')
plt.plot(np.arange(episodes), coop_ga_mean, label='GA under GA<->QL')

plt.ylabel('mean reward')
plt.xlabel('episodes/generations')
plt.legend(loc='best')
plt.savefig(outdata)
plt.show()
