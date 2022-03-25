
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from pathlib import Path

sns.set_context("talk")


folder = Path('E:/ALBERTO/expe20-21/repetition_def/results')
file = 'dataT_T.csv'
filename = folder / file
data = pd.read_csv(filename)#read datafile




#%% CALCULATE PROBABILITY FIRST SEEN AS REPETITION 

long = pd.melt(data, id_vars=['suje','DISTR'],
             value_vars=['2','3','4','5','6','7','8','9','10'],
        var_name='REPE', value_name='acc')#long format
long['acc'] = long['acc']/100# 0 to 1 values

long[['REPE','DISTR']] = long[['REPE','DISTR']].astype(int)
long.reset_index(inplace=True)
long = long.sort_values(['suje','DISTR','REPE'],axis=0)

# p-first= ((accuracy at repetition x) - (accuracy at repetition x-1)) /(1-accuracy at repetition x-1)
long['p_first'] = (long.acc-long.acc.shift(1)) / (1-long.acc.shift(1))
long['p_first'] = np.where(long['REPE']==2,long['acc'],long['p_first'])#p_first at Repetition 2 = accuracy at Repetition 2
long = long.replace([np.inf, -np.inf], np.nan)


#%% Figure
single = long[(long['DISTR']==1)&(long['REPE']<5)]
double = long[(long['DISTR']==2)&(long['REPE']<7)]
fig,ax = plt.subplots()
g = sns.pointplot(x = 'REPE', y = "p_first", ci = 68, color = 'slateblue',
                  ax = ax, data = single)

sns.pointplot(x = 'REPE', y = "p_first", color = 'r', linestyles = '--',
              ci=68, ax = ax, data = double)


sns.despine()
g.set(ylim = (0,1), xlim = (-0.5,4.2),
      xlabel = 'Number of Presentations',
      ylabel = 'Probability')
red_patch = mpatches.Patch(color='r', label='Two')
blue_patch = mpatches.Patch(color='slateblue', label='One')
plt.legend(handles = [blue_patch,red_patch],
            loc = 'upper right', 
           frameon = False, title = 'No. of distractors')
fig.set_size_inches(7,6)
#fig.savefig('p_first.png',dpi = 450)