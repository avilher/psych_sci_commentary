
import numpy as np
import pandas as pd
import os
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pathlib import Path
import pingouin as pg
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
#import bambi as bmb
import arviz as az
from scipy.stats.kde import gaussian_kde
from scipy.stats import halfnorm, norm
sns.set_context("talk")
def dprime(x,y):
    return stats.norm.ppf(x)-stats.norm.ppf(y)
data=pd.read_csv('thunell.csv')
#%%
def display_delta(model, x):
    # BFs based on density estimation (using kernel smoothing instead of spline)
   
    plt.style.use('default')
    t_delt = model.posterior['REPE'][:]
    my_pdf = gaussian_kde(t_delt)
    plt.plot(x, my_pdf(x), '--', color='black',lw=2.5, alpha=0.6, label='Posterior') # distribution function
    plt.plot(x, priorgaus(x), '-', color='black',lw=2.5, alpha=0.6, label='Prior')
    posterior = my_pdf(0)             # this gives the pdf at point delta = 0
             # height of order-restricted prior at delta = 0
   
    BF01      = posterior/priorgaus(0)
    print (f'the Bayes Factor is {BF01}')
    #print(prior)
    plt.plot([0, 0], [posterior, priorgaus(0)], 'k-', 
             [0, 0], [posterior, priorgaus(0)], 'ko', lw=1.5, alpha=1)
    plt.xlabel("Slope")
    plt.ylabel('Density')
    plt.xlim(-0.4,0.4)
    plt.legend(loc='upper left')
    plt.title('Bayes Factor = %.2f' %(BF01))
    plt.show()

def delta_halfnorm(model, x):
    # BFs based on density estimation (using kernel smoothing instead of spline)
   
    plt.style.use('default')
    t_delt = model.posterior['REPE'][:]
    my_pdf = gaussian_kde(t_delt)
    plt.plot(x, my_pdf(x), '--', color='black',lw=2.5, alpha=0.6, label='Posterior') # distribution function
    plt.plot(x, halfnorm.pdf(x,scale=0.1), '-', color='black',lw=2.5, alpha=0.6, label='Prior')
    posterior = my_pdf(0)             # this gives the pdf at point delta = 0
             # height of order-restricted prior at delta = 0
    
    BF01      = halfnorm.pdf(0,scale=0.1)/posterior
    print (f'the Bayes Factor is {BF01}')
    #print(prior)
    plt.plot([0, 0], [posterior, halfnorm.pdf(0,scale=0.1)], 'k-', 
             [0, 0], [posterior, halfnorm.pdf(0,scale=0.1)], 'ko', lw=1.5, alpha=1)
    plt.xlabel("Slope")
    plt.ylabel('Density')
    plt.xlim(-0.1,0.5)
    plt.legend(loc='upper left')
    plt.title('Bayes Factor = %.2f' %(BF01))
    plt.show()
#%%

data['1']=0
data=pd.melt(data, id_vars=['suje','DISTR'],
             value_vars=['1','2','3','4','5','6','7','8','9','10'],
        var_name='REPE', value_name='acc')
data['acc']=data['acc']/100
suumm=data
suumm['REPE']=suumm['REPE'].astype(int)
suumm['DISTR']=suumm['DISTR'].astype(int)
suumm.reset_index(inplace=True)
suumm=suumm.sort_values(['suje','DISTR','REPE'],axis=0)
suumm['acc2']=suumm['acc'].shift(1)
#suumm['acc2']=np.where(suumm['REPE']==2,suumm['acc3'].shift(1),suumm['acc2'])
suumm=suumm[suumm['REPE']>1]
suumm['tot']=1-suumm.acc2

suumm['accum']=suumm.acc-suumm.acc2
suumm['accum2']=suumm.accum/suumm.tot
#suumm['tot']=np.where(suumm['accum2']<0,1-suumm.acc,suumm['tot'])
#suumm['accum2']=suumm.accum/suumm.tot
#suumm['accum2']=np.where(suumm['REPE']==2,suumm['acc'],suumm['accum2'])## Uncoment to have it without chance
#suumm['accum2']=np.where(suumm['accum2']<0,0,suumm['accum2'])
suumm=suumm.replace([np.inf, -np.inf], np.nan)
#suumm['accum2']=suumm.accum2.fillna(1)
#suumm=suumm.dropna()
g=sns.catplot(x='REPE', y="accum2",hue='DISTR',ci=68,kind='point',data=suumm)
g.set(ylim=(-0.4,0.8),
      
       xlim=(-0.5,4.2),
      xlabel='REPETITIONS',
      ylabel='Probability')
#plt.plot([-0.5, 6], [0, 0], '--',color="black", lw=2)
#plt.ylim(0,1)
# suumm2=suumm[suumm.DISTR>2]
suumm['REPE']=suumm['REPE'].astype('category')
suumm['DISTR']=suumm['DISTR'].astype('category')
aov = pg.rm_anova(dv='accum2', within=['DISTR'],
                  subject='suje', data=suumm, detailed=True)
print(aov[['F','p-unc']].round(3))
suumm['REPE']=suumm['REPE'].astype(int)
suumm['DISTR']=suumm['DISTR'].astype('category')
#md = smf.mixedlm("accum2 ~ 1+REPE*DISTR", suumm, groups=suumm["suje"]).fit()
#print(md.wald_test_terms(skip_single=True))
gg=sns.catplot(x='DISTR', y='accum2',hue='DISTR',ci=68,kind='bar',data=suumm)
gg.set(ylim=(-0.4,0.6),
      xlabel='DISTRACTORS',
      ylabel='Probability')
plt.plot([-0.5,3], [0, 0], '--',color="black", lw=2)
#%%jacknife

data=pd.read_csv('thunell.csv')
sujes = list(data.suje.unique())
suumm2=pd.DataFrame()
for suj in sujes:
    data=pd.read_csv('thunell.csv')
    data=data[data.suje!=suj]
    data['1']=0
    data=pd.melt(data, id_vars=['suje','DISTR'],
                 value_vars=['1','2','3','4','5','6','7','8','9','10'],
            var_name='REPE', value_name='acc')
    data['acc']=data['acc']/100
    suumm=data
    suumm['REPE']=suumm['REPE'].astype(int)
    suumm['DISTR']=suumm['DISTR'].astype(int)
    suumm.reset_index(inplace=True)
    grouped=suumm.groupby(['DISTR','REPE']).agg({'acc':'mean'})
    grouped['suje']=suj
    grouped.reset_index(inplace=True)
    grouped=grouped.sort_values(['suje','DISTR','REPE'],axis=0)
    grouped['acc2']=grouped['acc'].shift(1)
    grouped=grouped[grouped['REPE']>1]
    grouped['accum']=grouped.acc-grouped.acc2
    grouped.reset_index(inplace=True)
    grouped['tot']=1-grouped.acc2
    grouped['accum2']=grouped.accum/grouped.tot
    grouped['suje']=suj
    suumm2=suumm2.append(grouped,ignore_index=True)
g=sns.catplot(x='REPE', y="accum2",hue='DISTR',ci=95,kind='point',data=suumm2)
g.set(ylim=(-0.4,0.8),
      xlabel='REPETITIONS',
      ylabel='Probability')


#%%
import pymc3 as pm
single=suumm[(suumm['DISTR']=='1')&(suumm['REPE']<5)]
single['suje']=single['suje'].astype('category')
group=single.suje.cat.codes.values
n_sujes = len(np.unique(group))
res = single.accum2.values
repes = single.REPE.values
with pm.Model() as bayes_rintercept:
    model_index = pm.DiscreteUniform('model_index', lower=0, upper=1)
    mu0 = pm.Normal('mu0', mu=np.mean(res), sigma=1)
    tau0 = pm.HalfStudentT('sd', sd=2, nu=10)
    a = pm.Normal('a', mu=mu0, sigma=tau0, shape=n_sujes)
    beta1 = pm.Normal('slope', mu=0, sigma=1)
    y_hat = pm.math.switch(pm.math.eq(model_index, 1),  a[group] + beta1*repes,
                               a[group] + beta1)


    sigma = pm.HalfStudentT('sd2', sd=2, nu=10)

    y = pm.Normal('y', mu=y_hat, sigma=sigma, observed=res)
    step1 = pm.Metropolis([model_index])
    trace = pm.sample(2000,[step1],tune=1000,cores=1,chains=2, progressbar=True)
burnin = 100  # posterior samples to discard
thin = 1  
model_idx_sample = trace['model_index'][burnin::thin]
## Compute the proportion of model_index at each value:
p_M1 = sum(model_idx_sample == 1) / len(model_idx_sample)
p_M2 = 1 - p_M1
bf=p_M2/p_M1

print('p(M1|D) = %.3f' % p_M1, 'p(M2|D) = %.3f' % p_M2, 'BF10 = %.3f'% bf)


#%%
import pymc3 as pm
single=suumm[(suumm['DISTR']=='1')&(suumm['REPE']<5)]
single['suje']=single['suje'].astype('category')
group=single.suje.cat.codes.values
n_sujes = len(np.unique(group))
res = single.accum2.values
repe = single.REPE.values
models=[]
traces=[]
repes=[1,repe]
for beta in repes:
    with pm.Model() as model:
        
        mu0 = pm.Normal('mu0', mu=np.mean(res), sigma=1)
        tau0 = pm.HalfStudentT('sd', sd=2, nu=10)
        a = pm.Normal('a', mu=mu0, sigma=tau0, shape=n_sujes)
        beta1 = pm.Normal('slope', mu=0, sigma=1)
        y_hat = a[group] + beta1*beta
    
    
        sigma = pm.TruncatedNormal('sigma', mu=0, sigma=100, lower=0)
    
        y = pm.Normal('y', mu=y_hat, sigma=sigma, observed=res)
        
        trace = pm.sample(1000,tune=1000,cores=1,chains=2, progressbar=True)
        models.append(model)
        traces.append(trace)

BF_smc =traces[1].marginal_likelihood / traces[0].marginal_likelihood
np.round(BF_smc)

#%%
single=suumm[(suumm['DISTR']=='1')&(suumm['REPE']<5)]
single['suje']=single['suje'].astype('category')
#md = smf.mixedlm("accum2 ~ 1+REPE", single, groups=single["suje"]).fit()
#print(md.wald_test_terms(skip_single=True))
formula="accum2 ~ REPE + (1|suje)"
prior = {"REPE":"medium","1|suje":"narrow"}
model1 = bmb.Model(data=single,formula=formula,priors=prior)

fitted1 = model1.fit(cores=1, draws=3000,chains=1)
prior_pred=model1.prior_predictive(draws=3000)

posterior_predictive = model1.posterior_predictive(fitted1)

az.plot_ppc(fitted1)
model1.plot_priors()
az.plot_trace(
    fitted1,
    var_names=['Intercept', 'REPE'],
    compact=True
);
t_delt = fitted1.posterior['REPE'][:]
p_delt = prior_pred.prior['REPE'][:]
my_pdf = gaussian_kde(t_delt)
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
#prior     = cauchy.pdf(0,0,1) 
priorgaus=     gaussian_kde(p_delt)   # height of order-restricted prior at delta = 0
bf01      = priorgaus(0)/posterior
plt.figure()
x = np.linspace(-0.5,0.5, 1000)
display_delta(fitted1, x)
delta_halfnorm(fitted1, x)
posterior = fitted1.posterior.stack({"draws":["chain", "draw"]})
interc = posterior['Intercept']
slope=posterior['REPE']
X = np.hstack([np.array([1] * len(np.arange(2, 5)))[:, None], np.arange(2, 5)[:, None]])
yhat = np.dot(X, np.vstack([interc, slope]))
slp=np.expand_dims(yhat,axis=-1).T
fig, ax = plt.subplots()

az.plot_hdi(X[:,1],slp)
az.plot_hdi(X[:,1],slp,hdi_prob=0.5)
sns.lineplot(x='REPE', y='accum2',ci=68,
             err_style='bars',ax=ax,data=single)
ax.set_xlim(1.7, 4.3);
ax.set_ylim(0, 0.8);
#%%
formula="accum2 ~  (1|suje)"
prior = {"1|suje":"narrow"}
model2 = bmb.Model(formula=formula,data=single,priors=prior)

fitted2 = model1.fit(cores=1,chains=1, 
                     draws=3000)
models_dict = {
    "full": fitted1,
    "null": fitted2}
df_compare = az.compare(models_dict,scale='deviance')
az.plot_compare(df_compare, insample_dev=True);
#%%
double=suumm[(suumm['DISTR']=='2')&(suumm['REPE']<7)]
double['suje']=double['suje'].astype('category')
model1 = bmb.Model(double)
prior = {"REPE":"medium","1|suje":"medium"}
fitted1 = model1.fit("accum2 ~ REPE + (1|suje)",
                     cores=1, draws=3000,chains=1,priors=prior)
prior_pred=model1.prior_predictive(draws=3000)
model1.plot_priors()
az.plot_trace(
    fitted1,
    var_names=['Intercept', 'REPE'],
    compact=True
);
t_delt = fitted1.posterior['REPE'][:]
p_delt = prior_pred.prior['REPE'][:]
my_pdf = gaussian_kde(t_delt)
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
#prior     = cauchy.pdf(0,0,1) 
priorgaus=     gaussian_kde(p_delt)   # height of order-restricted prior at delta = 0

plt.figure()
x = np.linspace(-0.5,0.5, 1000)
display_delta(fitted1, x)
delta_halfnorm(fitted1, x)
posterior = fitted1.posterior.stack({"draws":["chain", "draw"]})
interc = posterior['Intercept']
slope=posterior['REPE']
X = np.hstack([np.array([1] * len(np.arange(2, 7)))[:, None], np.arange(2, 7)[:, None]])
yhat = np.dot(X, np.vstack([interc, slope]))
slp=np.expand_dims(yhat,axis=-1).T
fig, ax = plt.subplots()

az.plot_hdi(X[:,1],slp)
az.plot_hdi(X[:,1],slp,hdi_prob=0.5)
sns.lineplot(x='REPE', y='accum2',ci=68,
             err_style='bars',ax=ax,data=double)
ax.set_xlim(1.7, 6.3);
ax.set_ylim(0, 0.8);
#%%

model2 = bmb.Model(double)
prior = {"1|suje":"narrow"}
fitted2 = model1.fit("accum2 ~  (1|suje)",cores=1,chains=1,
                     draws=3000,priors=prior)
models_dict = {
    "full": fitted1,
    "null": fitted2}
df_compare_single = az.compare(models_dict1,scale='deviance')
print(df_compare)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=False,gridspec_kw={'hspace':1})
az.plot_compare(df_compare_single, insample_dev=True,ax=ax1);
az.plot_compare(df_compare_double, insample_dev=True,ax=ax2);
ax1.set_title('One distractor')
ax2.set_title('Two distractors')
f.set_size_inches(7,4)
#%%
from bambi import Prior
single=suumm[(suumm['DISTR']=='1')&(suumm['REPE']<5)]
single['suje']=single['suje'].astype('category')
#md = smf.mixedlm("accum2 ~ 1+REPE", single, groups=single["suje"]).fit()
#print(md.wald_test_terms(skip_single=True))
model1 = bmb.Model(single)
my_favorite_prior = Prior('HalfNormal',sigma=0.1)
prior = {"REPE":my_favorite_prior,"1|suje":"medium"}
fitted1 = model1.fit("accum2 ~ REPE + (1|suje)",
                     cores=1, draws=3000,chains=1,priors=prior)
prior_pred=model1.prior_predictive(draws=3000)

posterior_predictive = model1.posterior_predictive(fitted1)

az.plot_ppc(fitted1)
model1.plot_priors()
az.plot_trace(
    fitted1,
    var_names=['Intercept', 'REPE'],
    compact=True
);
t_delt = fitted1.posterior['REPE'][:]
p_delt = prior_pred.prior['REPE'][:]
my_pdf = gaussian_kde(t_delt)
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
#prior     = cauchy.pdf(0,0,1) 
priorgaus=     gaussian_kde(p_delt)   # height of order-restricted prior at delta = 0
bf01      = priorgaus(0)/posterior
plt.figure()
x = np.linspace(-0.5,0.5, 1000)
display_delta(fitted1, x)
delta_halfnorm(fitted1, x)
posterior = fitted1.posterior.stack({"draws":["chain", "draw"]})
interc = posterior['Intercept']
slope=posterior['REPE']
X = np.hstack([np.array([1] * len(np.arange(2, 5)))[:, None], np.arange(2, 5)[:, None]])
yhat = np.dot(X, np.vstack([interc, slope]))
slp=np.expand_dims(yhat,axis=-1).T
fig, ax = plt.subplots()

az.plot_hdi(X[:,1],slp)
az.plot_hdi(X[:,1],slp,hdi_prob=0.5)
sns.lineplot(x='REPE', y='accum2',ci=68,
             err_style='bars',ax=ax,data=single)
ax.set_xlim(1.7, 4.3);
ax.set_ylim(0, 0.8);
model2 = bmb.Model(single)
prior = {"1|suje":"narrow"}
fitted2 = model2.fit("accum2 ~  (1|suje)",cores=1,chains=1, 
                     draws=3000,priors=prior)
models_dict1 = {
    "full": fitted1,
    "null": fitted2}
df_compare = az.compare(models_dict1,scale='deviance')
az.plot_compare(df_compare, insample_dev=True);
#%%
double=suumm[(suumm['DISTR']=='2')&(suumm['REPE']<7)]
double['suje']=double['suje'].astype('category')
model1 = bmb.Model(double)
my_favorite_prior = Prior('HalfNormal',sigma=0.1)
prior = {"REPE":my_favorite_prior,"1|suje":"medium"}
fitted1 = model1.fit("accum2 ~ REPE + (1|suje)",
                     cores=1, draws=3000,chains=1,priors=prior)
prior_pred=model1.prior_predictive(draws=3000)
model1.plot_priors()
az.plot_trace(
    fitted1,
    var_names=['Intercept', 'REPE'],
    compact=True
);
t_delt = fitted1.posterior['REPE'][:]
p_delt = prior_pred.prior['REPE'][:]
my_pdf = gaussian_kde(t_delt)
posterior = my_pdf(0)             # this gives the pdf at point delta = 0
#prior     = cauchy.pdf(0,0,1) 
priorgaus=     gaussian_kde(p_delt)   # height of order-restricted prior at delta = 0

plt.figure()
x = np.linspace(-0.1,0.8, 1000)
display_delta(fitted1, x)
delta_halfnorm(fitted1, x)
posterior = fitted1.posterior.stack({"draws":["chain", "draw"]})
interc = posterior['Intercept']
slope=posterior['REPE']
X = np.hstack([np.array([1] * len(np.arange(2, 7)))[:, None], np.arange(2, 7)[:, None]])
yhat = np.dot(X, np.vstack([interc, slope]))
slp=np.expand_dims(yhat,axis=-1).T
fig, ax = plt.subplots()

az.plot_hdi(X[:,1],slp)
az.plot_hdi(X[:,1],slp,hdi_prob=0.5)
sns.lineplot(x='REPE', y='accum2',ci=68,
             err_style='bars',ax=ax,data=double)
ax.set_xlim(1.7, 6.3);
ax.set_ylim(0, 0.8);

model2 = bmb.Model(double)
prior = {"1|suje":"narrow"}
fitted2 = model2.fit("accum2 ~  (1|suje)",cores=1,chains=1,
                     draws=3000,priors=prior)
models_dict = {
    "full": fitted1,
    "null": fitted2}
df_compare_double = az.compare(models_dict,scale='deviance')
print(df_compare_double)

az.plot_compare(df_compare_double, insample_dev=True);
#%%
import matplotlib.patches as mpatches
fig,ax=plt.subplots()
g=sns.pointplot(x='REPE', y="accum2",ci=68,color='slateblue',ax=ax,data=single)
flatui=['g','mediumblue','r','r','r']
sns.pointplot(x='REPE', y="accum2",color='r',linestyles='--',ci=None,ax=ax,data=double)

sns.pointplot(x='REPE', y="accum2",hue='REPE',palette=flatui,
              color='r',ci=68,ax=ax,data=double)
sns.despine()
g.set(ylim=(0,1),
      
       xlim=(-0.5,4.2),
      xlabel='Number of Presentations',
      ylabel='Probability')
grey_patch = mpatches.Patch(color='r', label='Two')
black_patch = mpatches.Patch(color='slateblue', label='One')
plt.legend(handles=[black_patch,grey_patch],
            loc='upper right', 
           frameon=False,title='No. of distractors')
fig.set_size_inches(7,6)
fig.savefig('thunell_first_rep.png',dpi=450)
#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
plt.style.use('seaborn-darkgrid')
x = np.linspace(0, 5, 200)
for b in [0.7, 1.0, 2.0]:
    pdf = st.cauchy.pdf(x, scale=b)
    plt.plot(x, pdf, label=r'$\beta$ = {}'.format(b))
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.legend(loc=1)
plt.show()