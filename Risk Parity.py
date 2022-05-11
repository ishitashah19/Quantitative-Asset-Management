#!/usr/bin/env python
# coding: utf-8

# <h1><center>QAM - PS2 Risk Parity- Ishita Shah</center></h1>

# In[1]:


import pandas as pd
import os
import numpy as np
import yfinance as yf
import datetime as dt
from datetime import timedelta
import time
import matplotlib.pyplot as plt
import wrds
from pandas.tseries.offsets import *
import pandas_datareader
import psycopg2
from random import *
from dateutil.relativedelta import *
import requests
from scipy.optimize import minimize


# In[6]:


os.chdir(r'C:\Users\ishit\OneDrive\Desktop\Ishita\Academics\UCLA - Spring Quarter 2022\QAM\Homework\HW 2')
os.getcwd()


# In[7]:


# Importing Data from wrds
# This is the folder where I will save the data
#data_folder = 'C:\\Users\\ishit\\OneDrive\\Desktop\\Ishita\\Academics\\UCLA - Spring Quarter 2022\\QAM\\Homework\\HW 1/data/'
id_wrds = 'ishitashah19'


# In[2]:


# import wrds
conn = wrds.Connection(wrds_username='ishitashah19')

# LOADING CRSP DATA FROM WRDS --  Bonds data
print('Load Bond Data')
bonds = conn.raw_sql("""
                      select kycrspid, mcaldt, tmretnua, tmtotout
                      from crspq.tfz_mth
                      """)
bonds['mcaldt']   = pd.DataFrame(bonds[['mcaldt']].values.astype('datetime64[ns]')) + MonthEnd(0)  #mcaldt will give the last day of the month
bonds = bonds.rename(columns={"mcaldt":"date","tmretnua":"ret","tmtotout":"me","kycrspid":"idCRSP"}).copy()

# LOADING CRSP DATA FROM WRDS -- T-Bill data
print('Load T-Bill Data')
rf = conn.raw_sql("""
                      select caldt, t30ret, t90ret
                      from crspq.mcti
                      """)
rf['caldt']   = pd.DataFrame(rf[['caldt']].values.astype('datetime64[ns]')) + MonthEnd(0)
rf = rf.rename(columns={"caldt":"date","t30ret":"rf30","t90ret":"rf90"}).copy()

# LOADING CRSP DATA FROM WRDS -- Equity Returns
print('Load Equity Returns Data')
crsp_raw = conn.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc, a.cfacshr
                      from crspq.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/1926' and '12/31/2021'
                      """)
crsp_raw[['permno', 'permco']] = crsp_raw[['permno', 'permco']].astype(int)
crsp_raw['date']   = pd.to_datetime(crsp_raw['date'],format='%Y-%m-%d',errors='ignore')
crsp_raw['date']   = pd.DataFrame(crsp_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)

# LOADING CRSP DATA FROM WRDS -- Deilsting Returns
print('Load Delisting Returns Data')
dlret_raw = conn.raw_sql("""
                     select permno, dlret, dlstdt, dlstcd
                     from crspq.msedelist
                     """)
dlret_raw.permno    = dlret_raw.permno.astype(int)
dlret_raw['dlstdt'] = pd.to_datetime(dlret_raw['dlstdt'])
dlret_raw['date']  = dlret_raw['dlstdt']+MonthEnd(0)

# LOADING CRSP DATA FROM WRDS -- CRSP VW market index (for comparison purposes)
print('Load VW CRSP market index Data')
mkt_csrp = conn.raw_sql("""
                      select date, VWRETD, totval
                      from crspq.msi
                      """)
mkt_csrp['date']   = pd.DataFrame(mkt_csrp[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
mkt_csrp = mkt_csrp.rename(columns={"vwretd":"mkt_crsp","totval":"mkt_crsp_mktcap"}).copy()

# Save data
mkt_csrp.to_pickle('mkt_csrp.pkl')
crsp_raw.to_pickle('crsp_raw.pkl')
dlret_raw.to_pickle('dlret_raw.pkl')
rf.to_pickle('rf.pkl')
bonds.to_pickle('bonds.pkl')

# Close connection to WRDS
conn.close()


# In[3]:


# Once the pickle files are loaded we would read the files
mkt_crsp=pd.read_pickle('mkt_csrp.pkl')
crsp_raw=pd.read_pickle('crsp_raw.pkl')
dlret_raw=pd.read_pickle('dlret_raw.pkl')
rf=pd.read_pickle('rf.pkl')
bonds=pd.read_pickle('bonds.pkl')


# In[4]:


bonds.head()


# ## $$Question 1$$

# Construct the equal-weighted bond market return, value-weighted bond market return, and lagged total bond market capitalization using CRSP Bond data 1. Your output should be from
# January 1926 to December 2021, at a monthly frequency.
# 

# In[6]:


print(bonds.head())
bonds.info()


# #### Now, we will begin our data cleaning process

# In[7]:


# Step 1 - Set idCRSP as string
bonds['idCRSP']=bonds['idCRSP'].astype('string')

# We will reset the index and sort the data by CRSP Id and date this will help in calculating lagged Market Cap
#bonds=bonds.sort_values(by=['idCRSP','date']).reset_index(drop=True).copy()

# Calculating lagged ME by grouping data by CRSP ID
bonds['lagme']=bonds.groupby('idCRSP')['me'].shift(1)

# Only considering returns which are not NAN
bonds=bonds[bonds['ret'].notna()].copy()


# In[8]:


bonds.head()


# In[9]:


sum1=bonds.groupby(['date'])['lagme'].sum()
weight=sum1.to_frame().rename(columns={'lagme':'sum'})
weight['id_num']=bonds.groupby('date')['idCRSP'].count()
bonds=bonds.merge(weight,how='left',on='date')
bonds['vweight']=bonds['lagme']/bonds['sum']
bonds['eweight']=1/bonds['id_num']
bonds['Bond_Vw_Ret']=bonds['ret']*bonds['vweight']
bonds['Bond_Ew_Ret']=bonds['ret']*bonds['eweight']


# In[10]:


bonds.head()


# Now that we have cleaned the data and calculated value weighted and equal weighted bond market returns - we will focus on calculating Bond Market Capitalization

# In[13]:


bonds_cleaned=bonds.groupby('date')['Bond_Vw_Ret'].sum().to_frame()
bonds_cleaned['Bond_Ew_Ret']=bonds.groupby('date')['Bond_Ew_Ret'].sum()
bonds_cleaned=bonds_cleaned.reset_index()
bonds_cleaned['year']=bonds_cleaned['date'].dt.year
bonds_cleaned['month']=bonds_cleaned['date'].dt.month
bonds_cleaned=bonds_cleaned.merge(sum1* 10**-3,how='left',on='date').rename(columns={'lagme':'Bond_lag_MV'})
PS2Q1 =bonds_cleaned[['year','month','Bond_lag_MV','Bond_Ew_Ret','Bond_Vw_Ret']]


# In[14]:


PS2Q1.head()


# ## $$Question 2$$

# Aggregate stock, bond, and riskless datatables. For each year-month, calculate the lagged market value and excess value-weighted returns for both stocks and bonds. Your output should be from January 1926 to December 2021, at a monthly frequency.

# In[16]:


load_data = True
wrds_id = 'ishitashah19'
conn = wrds.Connection(wrds_username=wrds_id)
#conn.create_pqpass_file()
#conn.close()

if load_data:
    conn = wrds.Connection(wrds_username=wrds_id)
    # Load CRSP returns and change variables format
    crsp_raw = conn.raw_sql("""
                            select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                            a.ret, a.retx, a.shrout, a.prc, a.cfacshr, a.cfacpr
                            from crspq.msf as a
                            left join crsp.msenames as b
                            on a.permno=b.permno
                            and b.namedt<=a.date
                            and a.date<=b.nameendt
                            where a.date between '01/01/1926' and '12/21/2021'
                            """)
    crsp_raw[['permno', 'permco']] = crsp_raw[['permno', 'permco']].astype(int)
    crsp_raw['date'] = pd.to_datetime(crsp_raw['date'],format='%Y-%m-%d',errors='ignore')
    #crsp_raw['date'] = pd.DataFrame(crsp_raw[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
    
    # Load CRSP Delisting returns and change variables format
    dlret_raw = conn.raw_sql("""
                            select permno, dlret, dlstdt, dlstcd
                            from crspq.msedelist
                            """)
    dlret_raw.permno = dlret_raw.permno.astype(int)
    dlret_raw['dlstdt'] = pd.to_datetime(dlret_raw['dlstdt'], format='%Y-%m-%d')
    dlret_raw['date']   = dlret_raw['dlstdt']+MonthEnd(0)
    conn.close()
    
  # Fama and French 3 Factors and change variables format
    pd.set_option('precision', 2)
    FF3 = pandas_datareader.famafrench.FamaFrenchReader('F-F_Research_Data_Factors', start='1926', end='2021')
    FF3 = FF3.read()[0]/100 #monthly data
    FF3.columns = 'MktRF','SMB','HML','RF'
    FF3['Mkt']  = FF3['MktRF'] + FF3['RF']
    FF3 = FF3.reset_index().rename(columns = {"Date" : "date"}).copy()
    FF3['date'] = pd.DataFrame(FF3[['date']].values.astype('datetime64[ns]')) + MonthEnd(0)
    
    #save pickle
    crsp_raw.to_pickle('crsp_raw.pkl')
    dlret_raw.to_pickle('dlret_raw.pkl')
    FF3.to_pickle('FF3.pkl')
else:
     #load pickle
    crsp_raw  = pd.read_pickle('crsp_raw.pkl')
    dlret_raw = pd.read_pickle('dlret_raw.pkl')
    FF3       = pd.read_pickle('FF3.pkl')
    


# In[61]:


#Merging Dataframes
mcrsp = pd.merge(crsp_raw, dlret_raw, how = 'left', on = ['permno', 'date'])
alternative_delisting = False
if alternative_delisting: # Alternative merging with delisting return
# Version 1: Set delisting date as the month following the last trading month or the month of delisting, whichever is earlier
    last_month = crsp_raw.groupby('permo')['date'].max().to_frame().reset_index().rename(columns={"date":"last_date"}).copy()
    last_month_month['last_date'] = last_month['last_date'] + timedelta(days=1) + MonthEnd(0)
    dlret_raw2 = dlret_raw.merge(last_month,on ='permno',how ='left').copy()
    dlret_raw2['date']= dlret_raw2[['date','last_date']].min(axis=1)
    dlret_raw2 = dlret_raw2.drop(['last_date'],axis = 1)
    
    # Version 2: Merge delisting returns & Crsp dataframes ('outer')
    mcrsp = pd.merge(crsp_raw, dlret_raw, how = 'outer', on =['permno','date'])
    mcrsp = mcrsp.sort_values(by = ['permno','date']).reset_index().drop('index', axis = 1).copy()
    cols = ['shrcd' ,'exchcd','permco']
    crsp[cols]= crsp.groupby('permno')[cols].ffill()

    # Keeping common shares (share codes 10 and 11) and major exchanges (exchage code 1, 2 and 3)
mcrsp = mcrsp[(mcrsp['shrcd'] == 10)| (mcrsp['shrcd']  == 11)]
mcrsp = mcrsp[(mcrsp['exchcd']== 1) | (mcrsp['exchcd'] == 2) | (mcrsp['exchcd'] == 3)]
#crsp = crsp[(crsp['exched']== 1)|(crsp['exchcd'] == 2)|(crsp['exchcd'] == 3)|\
# (crsp['exchcd'] == 31)|(crsp['exchcd'] == 32)|(crsp['exchcd'] == 33)]
mcrsp = mcrsp.drop(['exchcd','shrcd'], axis = 1)
mcrsp = mcrsp.copy()

# Compute returns including delisting returns
aux = mcrsp[(mcrsp['ret'].isna() & mcrsp['dlret'].isna())].index
mcrsp['dlret'] = mcrsp['dlret'].fillna(0)
mcrsp['ret']   = mcrsp['ret'].fillna(0)
mcrsp['ret']   = (1 + mcrsp['ret'])*(1+mcrsp['dlret'])-1
mcrsp.loc[aux,['ret']] = np.nan
mcrsp = mcrsp.drop(['dlstcd'], axis=1)
print(mcrsp.head())


# In[62]:


mcrsp['me'] = mcrsp['prc'].abs()*mcrsp['shrout']  # calculate market equity
mcrsp       = mcrsp.drop(['dlret', 'dlstdt', 'prc', 'permco', 'retx', 'shrout', 'cfacshr', 'cfacpr'], axis=1)

# Lagged Market Equity (to be used as weights)
mcrsp       = mcrsp.sort_values(by = ['permno', 'date']).reset_index().drop('index', axis=1).copy()
mcrsp['daten'] = mcrsp['date'].dt.year*12 + mcrsp['date'].dt.month
mcrsp['IsValidLag'] = mcrsp['daten'].diff(1) == 1 # Lag date has to be the lagged date
mcrsp.loc[mcrsp[mcrsp['permno'].diff(1) != 0].index, ['IsValidLag']] = False # Lagged data has to be the same security
mcrsp['Lme'] = mcrsp[['permno', 'me']].groupby('permno').shift(1)
mcrsp.loc[mcrsp[mcrsp['IsValidLag'] == False].index, ['Lme']] = np.nan
mcrsp = mcrsp.drop(['IsValidLag', 'daten'], axis=1)

#Choosing values where the lagged market value is greater than 0
mcrsp = mcrsp.loc[mcrsp['Lme'] > 0, :].copy()

# Dropping securities with missing lagged market equity values
mcrsp = mcrsp.sort_values(by = ['date', 'permno']).reset_index().drop('index', axis = 1).copy()

# Value Weighted returns - We won't be calculating equal weighted market returns as we would not need it for calculating Excess Returns
w = mcrsp.groupby(['date'])['Lme'].sum().to_frame().reset_index().rename(columns = {'Lme' : 'w'})  # This contains summed Lagged Market Equity for each month
w.loc[w['w'] == 0, 'w'] = np.nan
mcrsp = mcrsp.merge(w, how = 'left', on = 'date').copy()
mcrsp['w'] = mcrsp['Lme'] / mcrsp['w']
mcrsp['vwret'] = mcrsp['ret'] * mcrsp['w']
vwret = mcrsp.groupby(['date'])['vwret'].sum().to_frame().reset_index()

# Total market equity
mktcap = mcrsp.groupby(['date'])['me'].sum().to_frame().reset_index().rename(columns = {'me' : 'mktcap'})

# Equiweighted Portfolio
#mcrsp['permno_count'] = mcrsp.groupby('date')['permno'].count()
#mcrsp['ewret'] = 1/ mcrsp['permno_count']

# Now we would merge CRSP data with Fama- French Data
output_df = mktcap
output_df = pd.merge(output_df, vwret, how = 'outer', on = ['date'])
#output_df = pd.merge(output_df, FF3, how = 'outer', on = ['date'])
#output_df['excess_ret'] = output_df['vwret'] - output_df['RF']
#output_df = output_df.drop(['HML', 'SMB'], axis = 1)

# Keep only data available in both series
#output_df = output_df[output_df['vwret'].notna() & output_df['Mkt'].notna()].copy()


# In[73]:


output_df.head()
output_df['year'] = output_df['date'].dt.year
output_df['month'] = output_df['date'].dt.month
output_df.head()


# In[44]:


rf['year']=rf['date'].dt.year
rf['month']=rf['date'].dt.month


# In[74]:


rf.head()


# In[79]:


# Merging rf file
PS2Q2=PS2Q1.merge(rf,on=['year','month'],how='left')
PS2Q2 = PS2Q2.merge(output_df, on=['year','month'])
PS2Q2.head()


# In[80]:


PS2Q2['Bond_Excess_Vw_Ret']=PS2Q2['Bond_Vw_Ret']-PS2Q2['rf30']
PS2Q2['Stock_Excess_Vw_Ret'] = PS2Q2['vwret'] - PS2Q2['rf30']
PS2Q2.head()


# In[ ]:





# In[106]:


#PS2Q2 = pd.merge(PS2Q2, PS1Q1, on = ['year', 'month'])
PS2Q2 = PS2Q2.rename(columns = {'mktcap' : 'Stock_lag_MV'})
#PS2Q2 = PS2Q2.rename(columns = {'excess_ret' : 'Stock_Excess_Vw_Ret'})
PS2Q2 = PS2Q2[['year', 'month', 'Stock_lag_MV','Stock_Excess_Vw_Ret','Bond_lag_MV', 'Bond_Excess_Vw_Ret']]
PS2Q2.head()


# ## $$Question 3$$

# Calculate the monthly unlevered and levered risk-parity portfolio returns as defined by Asness, Frazzini, and Pedersen (2012).3 For the levered risk-parity portfolio, match the value-weighted portfolio’s ˆσ over the longest matched holding period of both. Your output should be from January 1926 to December 2021, at a monthly frequency.

# In[108]:


# Calculating excess returns of Value Weighted Portfolio and Excess returns of 60-40 Portfolio

ps2q3=PS2Q2.copy()
ps2q3['sw']=ps2q3['Stock_lag_MV']/(ps2q3['Stock_lag_MV']+ps2q3['Bond_lag_MV'])
ps2q3['bw']=ps2q3['Bond_lag_MV']/(ps2q3['Stock_lag_MV']+ps2q3['Bond_lag_MV'])
ps2q3['Excess_Vw_Ret']=ps2q3['bw']*ps2q3['Bond_Excess_Vw_Ret']+ps2q3['sw']*ps2q3['Stock_Excess_Vw_Ret']
ps2q3['Excess_60_40_Ret']=0.6*ps2q3['Stock_Excess_Vw_Ret']+0.4*ps2q3['Bond_Excess_Vw_Ret']
ps2q3=ps2q3.drop(['bw','sw'],axis=1)


# In[109]:


ps2q3


# In[110]:


# For Levered and Unlevered Risk Parity Portfolios
# Calculating Volatilities for both stocks and bonds in the portfolio

ps2q3['Stock_inverse_sigma_hat'] = 1/ps2q3['Stock_Excess_Vw_Ret'].rolling(36).std()
ps2q3['Bond_inverse_sigma_hat'] = 1/ps2q3['Bond_Excess_Vw_Ret'].rolling(36).std()

# We will calculate weights for Unlevered RP Portfolio
ps2q3['Unlevered_k']=1/(ps2q3['Stock_inverse_sigma_hat']+ps2q3['Bond_inverse_sigma_hat'])

# Calculating excess returns via Unlevered RP Portfolio
ps2q3['Excess_Unlevered_RP_Ret']=(ps2q3['Stock_inverse_sigma_hat']*ps2q3['Unlevered_k'])*ps2q3['Stock_Excess_Vw_Ret']+(ps2q3['Bond_inverse_sigma_hat']*ps2q3['Unlevered_k'])*ps2q3['Bond_Excess_Vw_Ret']


# In[111]:


# Considering Vol of Volume weighted excess returns

vw_vol = ps2q3['Excess_Vw_Ret'].std()


# In[101]:


def find_k(k):
    ps2q3['Excess_Levered_RP_Ret']=k*ps2q3['Stock_inverse_sigma_hat']*ps2q3['Stock_Excess_Vw_Ret']    +(ps2q3['Bond_inverse_sigma_hat']*k)*ps2q3['Bond_Excess_Vw_Ret']
    levervol=ps2q3['Excess_Levered_RP_Ret'].std()
    diff=abs(levervol-vw_vol)
    return diff

result = minimize(find_k, 0.01,method = 'Nelder-Mead')


# In[112]:


result


# In[113]:


k=result.x[0]
ps2q3['Levered_k']=k
ps2q3['Excess_Levered_RP_Ret']=k*ps2q3['Stock_inverse_sigma_hat']*ps2q3['Stock_Excess_Vw_Ret']    +(ps2q3['Bond_inverse_sigma_hat']*k)*ps2q3['Bond_Excess_Vw_Ret']


# In[114]:


ps2q3


# # $$ Question 4$$

# Replicate and report Panel A of Table 2 in Asness, Frazzini, and Pedersen (2012), exceptfor Alpha and t-stat of Alpha columns. Specifically, for all strategies considered, report the annualized average excess returns, t-statistic of the average excess returns, annualized volatility, annualized Sharpe Ratio, skewness, and excess kurtosis. Your sample should be from January 1930 to June 2010, at monthly frequency. Match the format of the table to the extent possible. Discuss the difference between your table and the table reported in the paper. It is zero? If not, justify whether the difference is economically negligible or not. What are the reasons a nonzero difference?

#  Restricting the data to the sample Period January 1930 to December 2021

# In[120]:


ps2q3=ps2q3[ps2q3['year']<=2010].copy().reset_index(drop=True)


# In[121]:


ps2q3 = ps2q3[ps2q3['year'] >= 1930].copy().reset_index(drop=True)
ps2q3


# In[122]:


df=len(ps2q3)
## mean
meanstock=ps2q3['Stock_Excess_Vw_Ret'].mean()*12
meanbond=ps2q3['Bond_Excess_Vw_Ret'].mean()*12
meanvw=ps2q3['Excess_Vw_Ret'].mean()*12
mean6040 = ps2q3['Excess_60_40_Ret'].mean()*12
meanunlevered = ps2q3['Excess_Unlevered_RP_Ret'].mean()*12
meanlevered = ps2q3['Excess_Levered_RP_Ret'].mean()*12


## Std
sdstock = ps2q3['Stock_Excess_Vw_Ret'].std()*np.sqrt(12)
sdbond = ps2q3['Bond_Excess_Vw_Ret'].std()*np.sqrt(12)
sdvw =ps2q3['Excess_Vw_Ret'].std()*np.sqrt(12)
sd6040 = ps2q3['Excess_60_40_Ret'].std()*np.sqrt(12)
sdunlevered = ps2q3['Excess_Unlevered_RP_Ret'].std()*np.sqrt(12)
sdlevered = ps2q3['Excess_Levered_RP_Ret'].std()*np.sqrt(12)

## T-stat
tstock=(ps2q3['Stock_Excess_Vw_Ret'].mean()*np.sqrt(df))/ps2q3['Stock_Excess_Vw_Ret'].std()
tbond=(ps2q3['Bond_Excess_Vw_Ret'].mean()*np.sqrt(df))/ps2q3['Bond_Excess_Vw_Ret'].std()
tvw = ps2q3['Excess_Vw_Ret'].mean()*np.sqrt(df)/ps2q3['Excess_Vw_Ret'].std()
t6040 = ps2q3['Excess_60_40_Ret'].mean()*np.sqrt(df)/ps2q3['Excess_60_40_Ret'].std()
tunlevered = ps2q3['Excess_Unlevered_RP_Ret'].mean()*np.sqrt(df)/ps2q3['Excess_Unlevered_RP_Ret'].std()
tlevered = ps2q3['Excess_Levered_RP_Ret'].mean()*np.sqrt(df)/ps2q3['Excess_Levered_RP_Ret'].std()

## Sharpe
srstock = meanstock/sdstock
srbond=meanbond/sdbond
srvw = meanvw/sdvw
sr6040 = mean6040/sd6040
srunlevered = meanunlevered/sdunlevered
srlevered = meanlevered/sdlevered

## Skewness
skewstock=ps2q3['Stock_Excess_Vw_Ret'].skew()
skewbond = ps2q3['Bond_Excess_Vw_Ret'].skew()
skewvw = ps2q3['Excess_Vw_Ret'].skew()
skew6040 = ps2q3['Excess_60_40_Ret'].skew()
skewunlevered = ps2q3['Excess_Unlevered_RP_Ret'].skew()
skewlevered = ps2q3['Excess_Levered_RP_Ret'].skew()

## Excess Kurtosis
kurstock = ps2q3['Stock_Excess_Vw_Ret'].kurtosis()-3
kurbond = ps2q3['Bond_Excess_Vw_Ret'].kurtosis()-3
kurvw = ps2q3['Excess_Vw_Ret'].kurtosis()-3
kur6040 = ps2q3['Excess_60_40_Ret'].kurtosis()-3
kurunlevered =  ps2q3['Excess_Unlevered_RP_Ret'].kurtosis()-3
kurlevered = ps2q3['Excess_Levered_RP_Ret'].kurtosis()-3


# In[123]:


d =  {'Annualized Mean': [meanstock,meanbond,meanvw,mean6040,meanunlevered,meanlevered], 't-stat of Annualized Mean': [tstock,tbond,tvw,t6040,tunlevered,tlevered],     'Annualized Std': [sdstock,sdbond,sdvw,sd6040,sdunlevered,sdlevered],'Annualized Sharpe Ratio': [rstock,rbond,rvw,r6040,runlevered,rlevered],      'Skewness': [skewstock,skewbond,skewvw,skew6040,skewunlevered,skewlevered],'Excess Kurtosis': [kurstock,kurbond,kurvw,kur6040,kurunlevered,kurlevered],
     }
df = pd.DataFrame(d, columns=d.keys(),index=pd.Index(['CRSP stocks', 'CRSP bonds', 'Value-weighted portfolio','60/40 portfolio', 'unlevered RP', 'levered RP']))
df

