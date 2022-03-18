
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import arviz as az
from bambi import Prior
from scipy.stats import halfnorm
from scipy.stats.kde import gaussian_kde
from pathlib import Path
sns.set_context("talk")

folder='E:/ALBERTO/expe20-21/repetition_def/psych_sci_commentary/psych_sci_commentary/'
fold=Path(folder)
datafold='data/'
datafolder = Path(datafold)
file = 'dataT_T.csv'
filename = folder / datafolder / file
data=pd.read_csv(filename)#read datafile

#%% CALCULATE PROBABILITY FIRST SEEN AS REPETITION 


long=pd.melt(data, id_vars=['suje','DISTR'],
             value_vars=['2','3','4','5','6','7','8','9','10'],
        var_name='REPE', value_name='acc')#long format
long['acc']=long['acc']/100# 0 to 1 values

long[['REPE','DISTR']]=long[['REPE','DISTR']].astype(int)
long.reset_index(inplace=True)
long=long.sort_values(['suje','DISTR','REPE'],axis=0)

# p-first= ((accuracy at repetition x) - (accuracy at repetition x-1)) /(1-accuracy at repetition x-1)
long['p_first']=(long.acc-long.acc.shift(1))/(1-long.acc.shift(1))
long['p_first']=np.where(long['REPE']==2,long['acc'],long['p_first'])#p_first at Repetition 2 = accuracy at Repetition 2
long=long.replace([np.inf, -np.inf], np.nan)



#%%
single=long[(long['DISTR']==1)&(long['REPE']<5)]
single['suje']=single['suje'].astype('category')
my_favorite_prior = Prior('HalfNormal',sigma=0.1)
prior = {"REPE":my_favorite_prior,"1|suje":"narrow"}


formula="p_first ~ REPE + (1|suje)"
model1 = bmb.Model(data=single,formula=formula,priors=prior)

fitted1 = model1.fit(cores=1, draws=3000,chains=1)


t_delt = fitted1.posterior['REPE'][:]
my_pdf = gaussian_kde(t_delt)
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
         # height of order-restricted prior at delta = 0

BF01 = halfnorm.pdf(0,scale=0.1)/posterior
print (f'the Bayes Factor is {BF01}')

prior2 = {"1|suje":"narrow"}

formula2="p_first ~ (1|suje)"
model2 = bmb.Model(data=single,formula=formula2,priors=prior2)

fitted2 = model2.fit(cores=1, draws=3000,chains=1)
models_dict1 = {
    "full": fitted1,
    "null": fitted2}
df_compare = az.compare(models_dict1,scale='deviance')
az.plot_compare(df_compare, insample_dev=True);
#%%
double=long[(long['DISTR']==2)&(long['REPE']<7)]
double['suje']=double['suje'].astype('category')
my_favorite_prior = Prior('HalfNormal',sigma=0.1)
prior = {"REPE":my_favorite_prior,"1|suje":"narrow"}


formula="p_first ~ REPE + (1|suje)"
model1 = bmb.Model(data=double,formula=formula,priors=prior)

fitted1 = model1.fit(cores=1, draws=3000,chains=1)


model1.plot_priors()
az.plot_trace(
    fitted1,
    var_names=['Intercept', 'REPE'],
    compact=True
);
t_delt = fitted1.posterior['REPE'][:]
my_pdf = gaussian_kde(t_delt)
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
         # height of order-restricted prior at delta = 0

BF01 = posterior/halfnorm.pdf(0,scale=0.1)
print (f'the Bayes Factor is {BF01}')

prior2 = {"1|suje":"narrow"}

formula2="p_first ~ (1|suje)"
model2 = bmb.Model(data=double,formula=formula2,priors=prior2)

fitted2 = model2.fit(cores=1, draws=3000,chains=1)
models_dict2 = {
    "full": fitted1,
    "null": fitted2}
df_compare2 = az.compare(models_dict2,scale='deviance')
az.plot_compare(df_compare2, insample_dev=True);

#%%
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                             gridspec_kw={'hspace':1})
az.plot_compare(df_compare, insample_dev=True,order_by_rank=False,ax=ax1);
az.plot_compare(df_compare2, insample_dev=True,ax=ax2);
ax1.set_title('One distractor')
ax2.set_title('Two distractors')
f.set_size_inches(7,4)