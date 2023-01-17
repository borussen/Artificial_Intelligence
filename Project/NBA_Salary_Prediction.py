#!/usr/bin/env python
# coding: utf-8

# 이전 두 시즌의 통계적 수치를 입력 변수로 사용하여 NBA 선수들의 연봉을 예측하는 인공지능 모델을 구축하는 프로젝트

# ## Import packages and data

# 해당 프로젝트에서는 2020/21 시즌 및 2021/22 시즌 NBA 선수별 통계 기록과 2022/23 시즌 NBA 선수 연봉을 데이터로 활용한다.

# In[1]:


# Data Manipulation
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import missingno as msno


# In[2]:


# Visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


# Machine Learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import model_selection, metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[4]:


# Other packages
from scipy.stats import norm
from scipy import stats


# In[5]:


# Import data
df_salary = pd.read_csv('./file/NBA_Player_Salary.csv')
df_stats2021 = pd.read_csv('./file/NBA_Player_Stats(2021).csv')
df_stats2122 = pd.read_csv('./file/NBA_Player_Stats(2122).csv')
df_list = [df_stats2021, df_stats2122, df_salary]


# ## 1. Data Cleaning

# 데이터셋으로 지정된 파일 각각의 처음 다섯 행을 살펴본다.

# In[6]:


# Salary
df_salary.head()


# In[7]:


# Season Stats 20/21
df_stats2021.head()


# In[8]:


# Season Stats 21/22
df_stats2122.head()


# 2022/23 시즌 선수들의 연봉을 예측하는 것이 프로젝트의 목표이므로 해당 시즌의 연봉에 관한 열을 가공한다.

# In[9]:


# Rename salary column
df_salary = df_salary.rename(columns = {'2022-23': 'Salary 22/23'})

# Transform salary to 1000
df_salary['Salary 22/23'] = df_salary['Salary 22/23']/1000


# 시즌별 선수 통계 데이터셋에서 동일 플레이어에 대한 중복 행을 삭제한다.
# * 프로젝트의 목표인 2022/23 시즌 연봉을 구하는 데 있어서 선수가 한 시즌 동안 어느 팀에서 뛰었는 지는 중요한 요소가 아니다.<br>그러므로 해당 시즌 동안 여러 팀에서 뛴 선수들의 행을 하나로 통합한다.  

# In[10]:


# As total stats always is in the top row we can simply use the drop_duplicates function
df_stats2021 = df_stats2021.drop_duplicates(['Player'])
df_stats2122 = df_stats2122.drop_duplicates(['Player'])


# ### Merge Datasets

# 각 시즌별 선수 통계 데이터셋의 모든 열에 해당하는 연도를 할당한다.

# In[11]:


# Add season year to corresponding columns
columns_renamed = [s + ' 20/21' for s in list(df_stats2021.columns)]
df_stats2021.columns = list(df_stats2021.columns)[:3] + columns_renamed[3:]

columns_renamed = [s + ' 21/22' for s in list(df_stats2122.columns)]
df_stats2122.columns = list(df_stats2122.columns)[:3] + columns_renamed[3:]

# Delete Pos column from 17/18 df; we need it only once
df_stats2021 = df_stats2021.drop('Pos', axis = 1)


# In[12]:


# Merge datasets
df_stats = df_stats2021.merge(df_stats2122, how = 'outer',left_on = ['Player'],right_on = ['Player'])
df = df_stats.merge(df_salary, how = 'outer', left_on = ['Player'],right_on = ['Player'])

df.head()


# 데이터셋의 각각의 열에 해당하는 데이터 타입을 확인한다.

# In[13]:


df.dtypes


# ### Drop unnecessary columns

# 앞으로의 프로젝트 과정에서 사용되지 않는 일부 열을 제거한다.

# In[14]:


# Columns of dataset
df.columns


# In[15]:


# Drop unnecessary columns
df = df.drop(['Rk_x', 'Rk_y', 'Rk', 'Tm', '2023-24', '2024-25', '2025-26', '2026-27', '2027-28',
              'Signed Using', 'Guaranteed'], axis = 1)


# ### Missing values

# 각 열에 존재하는 결측값을 확인한다.

# In[16]:


# Number of missing values for each column
df.isnull().sum()


# 20/21 시즌과 21/22 시즌에 대하여 Salary나 Stat 정보가 없는 행은 제거한다.
# 
# * 선수들 중 일부는 20/21 시즌이나 21/22 시즌에 출전하지 않았으며, 몇몇 선수들은 21/22 시즌에 대한 계약을 맺지 못했다.<br> 이러한 경우는 연봉을 예측하는 데 있어서 도움이 되지 못하기 때문에 결측값으로 판단하여 제거하는 것이 합리적이다.

# In[17]:


# Drop rows with NaN
# Create Dataframe without stats for season 20/21
df1 = df.dropna(subset = ['Salary 22/23', 'PTS 21/22', 'eFG% 21/22'])
df1 = df1.reset_index()
columns = list(df1.columns)
for i in columns:
    if '20/21' in i:
        df1 = df1.drop([i], axis = 1)
df1 = df1.reset_index()

# Create Dataframe with stats for season 20/21
df2 = df.dropna(subset = ['Salary 22/23', 'PTS 21/22', 'eFG% 21/22', 'PTS 20/21', 'eFG% 20/21'])
df2 = df2.reset_index()


# 데이터셋 내에 존재하는 포지션을 확인한다.

# In[18]:


# Get unique positions
print(df1.Pos.unique())
print(df2.Pos.unique())


# 5개의 포지션: PG, SG , PF, SF, C으로 간소화하여 분류한다.

# In[19]:


# Replace duplicate positions with first position.
df1 = df1.replace({'SG-PG': 'SG', 'PG-SG': 'PG', 'SF-SG': 'SF', 'SG-SF':'SG', 'C-PF': 'C', 'PF-C':'PF'})
df2 = df2.replace({'SG-PG': 'SG', 'PG-SG': 'PG', 'SF-SG': 'SF', 'SG-SF':'SG', 'C-PF': 'C', 'PF-C':'PF'})


# 주요 기록들에 대한 absolute growth를 얻기 위해서 데이터 프레임(df2)에 대한 20/21 시즌의 통계를 사용한다.

# In[20]:


# Get absolute growth
# List of stats of which we want the growth
list_growth = ['eFG%','TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# Add absolute growth columns
for i in list_growth:
    df2[i + ' +-'] = df2[i + ' 21/22'] - df2[i + ' 20/21']
    
# Drop 17/18 columns
columns = list(df2.columns)
for i in columns:
    if '20/21' in i:
        df2 = df2.drop([i], axis = 1)


# In[21]:


print(df1.shape)
print(df2.shape)


# 20/21 시즌과 21/22 시즌에 대한 선수별 통계 기록과 22/23 시즌의 연봉 정보를 가지는 행은 총 340개가 존재한다.<br>
# 21/22 시즌의 선수별 통계 기록과 22/23 시즌의 연봉 정보를 가지고 있는 행은 409개로, 69개의 추가 행이 있다. 

# ## 2. Exploring our dataset

# ### Target variable인 연봉 분석 (1000 단위)

# In[22]:


# Setup dataframe
df_sal = df1[['Player', 'Salary 22/23']]
df_sal.sort_values(by = 'Salary 22/23', ascending = False, inplace = True)

# Create barchart
sns.catplot(x = 'Player', y = 'Salary 22/23', kind = 'bar', data = df_sal.head()).set(xlabel = None)
plt.title('Players with highest salary (in 1000)')
plt.ylim([42000, 48000])
plt.xticks(rotation = 90)


# 연봉 관련 주요 통계적 요소에 대한 정보 요약

# In[23]:


# Statistics summary
df1['Salary 22/23'].describe()


# 연봉 분포(salary distribution)에 대한 히스토그램 모양 확인

# In[24]:


# Histogram
sns.distplot(df1['Salary 22/23'])


# 히스토그램을 통해 다음을 확인할 수 있다.
# 
# 1. 연봉이 분산되어 있음을 나타내는 큰 표준 편차
# 2. 선수들의 연봉이 정규 분포(normal distribution)를 따르지 않음
# 3. 우편향 분포 (right-skewed distribution)

# ### Analysis of most important player stats

# NBA 선수들의 주요 기록(경기당 득점, 어시스트, 스틸, 리바운드)에 대해 살펴본다.

# 우선 각 지표의 리더를 확인한다.

# In[25]:


# Setup dataframes
df_pts = df1[['Player', 'PTS 21/22']]
df_pts.sort_values(by = 'PTS 21/22', ascending = False, inplace = True)
df_ast = df1[['Player', 'AST 21/22']]
df_ast.sort_values(by = 'AST 21/22', ascending = False, inplace = True)
df_stl = df1[['Player', 'STL 21/22']]
df_stl.sort_values(by = 'STL 21/22', ascending = False, inplace = True)
df_trb = df1[['Player', 'TRB 21/22']]
df_trb.sort_values(by = 'TRB 21/22', ascending = False, inplace = True)

# Set up figure
f, axes = plt.subplots(2, 2, figsize=(20, 15))
sns.despine(left=True)

# Create barcharts
sns.barplot(x = 'PTS 21/22', y = 'Player', data = df_pts.head(), color = "b", ax = axes[0, 0]).set(ylabel = None)
sns.barplot(x = 'AST 21/22', y = 'Player', data = df_ast.head(), color = "r", ax = axes[0, 1]).set(ylabel = None)
sns.barplot(x = 'STL 21/22', y = 'Player', data = df_stl.head(), color = "g", ax = axes[1, 0]).set(ylabel = None)
sns.barplot(x = 'TRB 21/22', y = 'Player', data = df_trb.head(), color = "m", ax = axes[1, 1]).set(ylabel = None)


# 히스토그램을 이용하여 각 지표의 분산을 확인한다. 

# In[26]:


# Set up figure
f, axes = plt.subplots(2, 2, figsize=(20, 15))
sns.despine(left=True)

# Histograms
sns.distplot(df1['PTS 21/22'], color = "b", ax = axes[0, 0])
sns.distplot(df1['AST 21/22'], color = "r", ax = axes[0, 1])
sns.distplot(df1['STL 21/22'], color = "g", ax = axes[1, 0])
sns.distplot(df1['TRB 21/22'], color = "m", ax = axes[1, 1])


# 농구의 중요 통계적 수치 역시 대부분 우편향 분포(right-skewed distribution)임을 확인할 수 있다. 

# ### Relationship with possible features

# 22/23 시즌 연봉과 21/22 시즌 경기당 득점, 어시스트, 스틸, 리바운드 사이의 관계를 각각 살펴본다.

# In[27]:


# Set up figure
f, axes = plt.subplots(2, 2, figsize=(20, 15))

# Regressionplot
sns.regplot(x = df1['PTS 21/22'], y = df1['Salary 22/23'], color="b", ax=axes[0, 0])
sns.regplot(x = df1['AST 21/22'], y = df1['Salary 22/23'], color="r", ax=axes[0, 1])
sns.regplot(x = df1['STL 21/22'], y = df1['Salary 22/23'], color="g", ax=axes[1, 0])
sns.regplot(x = df1['TRB 21/22'], y = df1['Salary 22/23'], color="m", ax=axes[1, 1])


# 선택된 모든 통계적 수치들과 연봉 사이에는 positive한 선형 관계가 있다.

# In[28]:


# Relationship with effecitve field goal percentage
sns.regplot(x = df1['eFG% 21/22'], y = df1['Salary 22/23'])


# 농구에서는 2점슛과 3점슛 간의 차이가 있기 때문에 normal field goal percentage 대신 effective field goal percentage를 선택했다.<br>
# 그러나 위 그래프에서 확인할 수 있듯이 postive한 선형 관계를 관찰하기 어렵다.<br>
# (eFG%가 높으면서 연봉이 낮은 선수들은 있지만, eFG%가 낮으면서 높은 연봉을 받는 선수는 존재하지 않는다.)

# In[29]:


# Relationship with minutes played per game
sns.regplot(x = df1['MP 21/22'], y = df1['Salary 22/23'])


# 경기당 경기 시간과 연봉 사이에는 exponential relationship으로 보이는 positive relationship이 존재한다.<br> 
# 그러나 좋은 기록을 가지는 선수는 다음 시즌에 높은 연봉과 많은 경기시간을 보장 받을 가능성이 높기 때문에 해당 관계의 분석에 신중할 필요가 있다.<br> 
# 즉, 이러한 postive relationship은 기록에서 기반한 것으로 판단이 가능하다.

# In[30]:


# Relationship with age
sns.regplot(x = df1['Age 21/22'], y = df1['Salary 22/23'])


# 선수의 나이와 연봉 사이에는 선형적인 관계가 없다.

# In[31]:


# Relationship with Position
sns.boxplot(x = 'Pos', y = 'Salary 22/23', data = df1, order = ['PG', 'SG', 'SF', 'PF', 'C'])


# ### Correlation matrix

# 지금까지의 분석은 직관에 따라 연봉을 결정하는 데 있어 중요하다고 생각되는 요소들에 기반을 두고 진행했다.<br>
# 좀 더 객관적인 분석을 진행하고 변수 간의 관계에 대하여 완벽한 개요를 얻기 위해서 히트맵을 활용한다.

# In[32]:


sns.set(style = "white")
cor_matrix = df1.loc[:, 'Age 21/22': 'Salary 22/23'].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))

plt.figure(figsize = (15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})


# 히트맵을 통한 분석
# 
# 1. 다중 공선성: 정사각형의 색깔이 붉을수록 높은 상관 관계를 가지는 변수들로, 예측 모델에 거의 동일한 정보를 제공하고 추정에 대한 분산을 증가시킨다. 회귀 분석은 특징들 간의 독립성을 전제로 하기 때문에 올바른 회귀 분석을 위해서는 이러한 쌍을 제거해야한다. 다중 공선성은 분산팽창요인(VIF, Variance Inflation Factor) 이라는 계수로 평가할 수 있으며, 일반적으로 VIF 계수가 10~15 정도를 넘으면 다중 공선성 문제가 발생했다고 판단한다.
# 
# 
# 2. 22/23 시즌 연봉과의 상관관계: 앞서 분석했듯이, 연봉과 네 가지의 주요 통계 요소들 사이에는 선형적인 관계가 존재한다. 그러나 이 밖에도 추가적으로 고려해야 할 다른 변수들도 있음을 확인할 수 있다.
# 
# 연봉과 경기당 플레이 시간(MP, minutes played)이나 선발 출전 횟수(GS, games started)의 상관관계를 분석할 때는 신중해야 하는데, 단순하게 선발 출전을 많이 하는 선수가 더 많은 돈을 번다고 할 수 없기 때문이다. 공격과 수비에 대한 통계를 참고하여 코치는 선수가 경기에 얼마나 오래 출전할 지를 결정하고, 이에 따라 다음 시즌의 연봉 역시 결정되므로 인과관계라고 하기는 어렵다. 다만 총 게임 수(G)는 모델에 활용이 가능하다. 이 통계는 선수의 취약성과 연봉의 관계를 측정할 수 있다. 또한 높은 상관관계로 인해 TOV(Turnover)가 많을수록 연봉이 높아진다고 말하기는 어렵다.  공격력과 수비력이 좋은 선수는 더 많은 경기 시간과 더 높은 연봉을 보장 받는다. 그렇기 때문에 연봉이 높은 좋은 선수는 경기 시간이 많아지고, 평균적으로 턴오버 역시 더 많을 수 밖에 없다.

# In[33]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

V = df1[['Age 21/22', 'G 21/22', 'GS 21/22', 'MP 21/22', 'FG 21/22', 'FGA 21/22',
         'FG% 21/22', '3P 21/22', '3PA 21/22', '2P 21/22', '2PA 21/22', '2P% 21/22', 
         'eFG% 21/22', 'FT 21/22', 'FTA 21/22', 'ORB 21/22', 'DRB 21/22', 'TRB 21/22', 
         'AST 21/22', 'STL 21/22', 'BLK 21/22', 'TOV 21/22', 'PF 21/22', 'PTS 21/22', 
         'PER 21/22', 'TS% 21/22', '3PAr 21/22', 'FTr 21/22', 'ORB% 21/22', 'DRB% 21/22',
         'TRB% 21/22', 'AST% 21/22', 'STL% 21/22', 'BLK% 21/22', 'TOV% 21/22', 'USG% 21/22',
         'OWS 21/22', 'DWS 21/22', 'WS 21/22', 'WS/48 21/22' ,'OBPM 21/22', 'DBPM 21/22',
         'BPM 21/22', 'VORP 21/22']]

# 특징마다의 VIF 계수를 출력합니다.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(V.values, i) for i in range(V.shape[1])]
vif["features"] = V.columns
vif.round(1)


# 다중 공선성 문제가 없는 적절한 특징들을 선정한다.
# 
# * 선정 과정
# 1. VIF 계수가 높은 특징을 우선적으로 제거한다. 단, (FG, FG%)와 같이 유사한 특징 중에서는 하나만을 제거한다.
# 
# 2. 다시 다중 공선성을 검증한다.<br>
# (VIF 계수가 비정상적으로 높은 특징을 제거해주면, 다른 특징들의 공선성이 감소하는 것을 확인할 수 있다.)
# 
# 3. 여전히 VIF 계수가 높은 특징들을 제거한다.

# In[34]:


V = df1[['Age 21/22', 'G 21/22', 'GS 21/22', 'MP 21/22', 'FGA 21/22','3P 21/22', '2PA 21/22', 
         'FT 21/22', 'DRB 21/22', 'AST 21/22', 'STL 21/22', 'BLK 21/22', 'TOV 21/22', 'PF 21/22', 'PTS 21/22', 
         'PER 21/22', 'TS% 21/22', '3PAr 21/22', 'FTr 21/22', 'AST% 21/22', 'STL% 21/22', 
         'BLK% 21/22', 'TOV% 21/22', 'USG% 21/22','WS/48 21/22' ,'OBPM 21/22', 'VORP 21/22']]

# 특징마다의 VIF 계수를 출력합니다.
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(V.values, i) for i in range(V.shape[1])]
vif["features"] = V.columns
vif.round(1)


# ### What about the relationship between the absolute changes and salary?

# 20/21 시즌과 21/22 시즌의 성적 변화가 선수들의 연봉을 결정하는 데 영향을 미치는 지 확인한다.  

# In[35]:


cor_matrix = df2.loc[:, ['eFG% +-','TRB +-', 'AST +-', 'STL +-', 'BLK +-', 'TOV +-', 'PF +-', 'PTS +-', 
                        'Salary 22/23']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(cor_matrix, dtype = np.bool))

plt.figure(figsize = (10, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)

sns.heatmap(cor_matrix, mask = mask, cmap = cmap, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})


# Absolute change와 연봉 간에는 선형적인 관계는 없어 보인다.<br> 그러므로 예측 모델을 구현 하는 데 있어서 absolute changes와 20/21 시즌의 통계 수치를 제외한다.

# ## 3. Data Preparation

# ### Feature Scaling

# 특징들 간의 단위를 맞추기 위해서 StandardScaler를 이용한 Scaling을 진행한다.

# In[36]:


# Define the function scaling for each feature
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
    return df


# In[37]:


# Scaling for each features
scale_columns = ['Age 21/22', 'G 21/22', 'GS 21/22', 'MP 21/22', 'FG 21/22', 'FGA 21/22',
         'FG% 21/22', '3P 21/22', '3PA 21/22', '2P 21/22', '2PA 21/22', '2P% 21/22', 
         'eFG% 21/22', 'FT 21/22', 'FTA 21/22', 'ORB 21/22', 'DRB 21/22', 'TRB 21/22', 
         'AST 21/22', 'STL 21/22', 'BLK 21/22', 'TOV 21/22', 'PF 21/22', 'PTS 21/22', 
         'PER 21/22', 'TS% 21/22', '3PAr 21/22', 'FTr 21/22', 'ORB% 21/22', 'DRB% 21/22',
         'TRB% 21/22', 'AST% 21/22', 'STL% 21/22', 'BLK% 21/22', 'TOV% 21/22', 'USG% 21/22',
         'OWS 21/22', 'DWS 21/22', 'WS 21/22', 'WS/48 21/22' ,'OBPM 21/22', 'DBPM 21/22',
         'BPM 21/22', 'VORP 21/22']
df1 = standard_scaling(df1, scale_columns)
df1.head(5)


# ### Define target variable and features

# In[38]:


y = df1.loc[:, 'Salary 22/23']

x = df1.loc[:, ['Pos', 'Age 21/22', 'G 21/22', 'GS 21/22', 'MP 21/22', 'FGA 21/22','3P 21/22', '2PA 21/22', 
         'FT 21/22', 'DRB 21/22', 'AST 21/22', 'STL 21/22', 'BLK 21/22', 'TOV 21/22', 'PF 21/22', 'PTS 21/22', 
         'PER 21/22', 'TS% 21/22', '3PAr 21/22', 'FTr 21/22', 'AST% 21/22', 'STL% 21/22', 
         'BLK% 21/22', 'TOV% 21/22', 'USG% 21/22','WS/48 21/22' ,'OBPM 21/22', 'VORP 21/22']] 

print(x.shape)
print(y.shape)


# ### One-Hot encoding for feature position

# 범주형 변수를 다루기 위해서 One-Hot encoder를 사용한다.

# In[39]:


# Instantiate OneHotEncoder
ohe = OneHotEncoder(categories = [['PG', 'SG', 'SF', 'PF', 'C']])

# Apply one-hot encoder
x_ohe = pd.DataFrame(ohe.fit_transform(x['Pos'].to_frame()).toarray())

# Get feature names
x_ohe.columns = ohe.get_feature_names(['Pos'])

# One-hot encoding removed index; put it back
x_ohe.index = x.index

# Add one-hot encoded columns to numerical features and remove categorical column
x = pd.concat([x, x_ohe], axis=1).drop(['Pos'], axis=1)

# How does it look like?
x.head()


# ### Split Data into train and test

# 회귀 분석을 위한 학습, 테스트 데이터셋 분리한다.

# In[40]:


# Split data using train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ### Normalise y

# 앞서 종속 변수(dependent variable)가 정규 분포를 근사적으로 따르지 않는 것을 확인했다.<br>
# 그러므로 머신러닝 모델을 구현하기 위해서 정규화를 해준다.

# In[41]:


#Apply cube-root transformation
y_train = pd.DataFrame(np.cbrt([y_train])).T
y_test = pd.DataFrame(np.cbrt([y_test])).T
y = pd.DataFrame(np.cbrt([y])).T

#transformed histogram and normal probability plot
f, axes = plt.subplots(1, 2, figsize = (10, 5), sharex = True)
sns.distplot(y_train, color = "skyblue", fit = norm, ax = axes[0], axlabel = "y_train")
sns.distplot(y_test, color = "olive",fit = norm, ax = axes[1], axlabel = "y_test")
#sns.distplot(y, color = "olive",fit = norm, axlabel = "y")


# ## 4. Building Machine Learning Models

# ### Basic machine learning algorithms

# Model 평가 기준으로 RMS(Root-Mean-Squared Error)와 R-Squared를 활용한다.
# 
# 1. RMSE score: 실제값과 예측값의 차이를 절대값으로 나타낸 것으로, 해당 값이 높을수록 예측이 부정확하다는 뜻이다.
# 
# 
# 2. R-Squared score: 회귀 분석으로 추정한 모델이 주어진 데이터를 얼마나 잘 설명하는지 나타내는 점수로, 데이터를 잘 설명하는 모델일수록 1에 가깝다.

# In[42]:


# Function which uses an algorithm as input and returns the desired accuracy metrics and some predictions

def alg_fit(alg, x_train, y_train, x_test, name, y_true, df, mse, r2):
    
    # Model selection
    mod = alg.fit(x_train, y_train)
    
    # Prediction
    y_pred = mod.predict(x_test)
    
    # Accuracy
    acc1 = round(mse(y_test, y_pred), 4)
    acc2 = round(r2(y_test, y_pred), 4)
    
    # Accuracy table
    x_test['y_pred'] = mod.predict(x_test)
    df_acc = pd.merge(df, x_test, how = 'right')
    x_test.drop(['y_pred'], axis = 1, inplace = True)
    df_acc = df_acc[[name, y_true, 'y_pred']]
    df_acc.sort_values(by = y_true, ascending = False, inplace = True)
    df_acc['y_pred'] = df_acc['y_pred']**3
    
    return y_pred, acc1, acc2, df_acc


# ### Linear Regression

# In[43]:


# Linear Regression
y_pred_lin, mse_lin, r2_lin, df_acc_lin = alg_fit(LinearRegression(), x_train, y_train, x_test, 'Player', 'Salary 22/23', 
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_lin), 4))
print("R-squared: %s" % r2_lin)
df_acc_lin.head(10)


# ### Ridge Regression

# 선형 회귀(Linear Regression) 모델은 다중 공선성 문제를 가지기 때문에 이에 대한 적합한 해결책을 찾아야 한다.<br>
# 다중 공선성 문제는 부정확한 추정을 유발하고 큰 표준 오차를 가지게 한다.<br> 
# Ridge regression은 람다 값을 조절해 추정에 개선된 효율성을 제공하여 해당 문제를 완화한다.

# In[44]:


# Ridge Regression
y_pred_rid, mse_rid, r2_rid, df_acc_rid = alg_fit(Ridge(alpha = 1), x_train, y_train, x_test, 'Player', 'Salary 22/23',
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_rid), 4))
print("R-squared: %s" % r2_rid)
df_acc_rid.head(10)


# ### Lasso Regression

# Lasso Regression는 Ridge Regression과 개념적으로 매우 유사하다. 0이 아닌 계수에 대해 패널티를 추가로 부여하며, Ridge Regression과 다르게 계수의 제곱합이 아닌 계수의 절대값으로 제한한다.

# In[45]:


# Lasso Regression
y_pred_las, mse_las, r2_las, df_acc_las = alg_fit(Lasso(alpha = 0.001), x_train, y_train, x_test, 'Player', 'Salary 22/23',
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_las), 4))
print("R-squared: %s" % r2_las)
df_acc_las.head(10)


# ### Cross Validation

# Cross Validation을 통해 보다 정교한 정확도 측정을 제공한다.

# In[46]:


def alg_fit_cv(alg, x, y, mse, r2):
    
    # Cross validation
    cv = KFold(shuffle = True, random_state = 0, n_splits = 5)
    
    # Accuracy
    scores1 = cross_val_score(alg, x, y, cv = cv, scoring = mse)
    scores2 = cross_val_score(alg, x, y, cv = cv, scoring = r2)
    acc1_cv = round(scores1.mean(), 4)
    acc2_cv = round(scores2.mean(), 4)
    
    return acc1_cv, acc2_cv


# In[47]:


# Linear Regression

mse_cv_lin, r2_cv_lin = alg_fit_cv(LinearRegression(), x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_lin*-1), 4))
print("R-squared: %s" % r2_cv_lin)


# In[48]:


# Ridge Regression
mse_cv_rid, r2_cv_rid = alg_fit_cv(Ridge(alpha = 23), x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_rid*-1), 4))
print("R-squared: %s" % r2_cv_rid)


# In[49]:


# Lasso Regression
mse_cv_las, r2_cv_las = alg_fit_cv(Lasso(), x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_las*-1), 4))
print("R-squared: %s" % r2_cv_las)


# ### Advanced models
# 

# ### LightGBM Regressor

# In[50]:


# LightGBM Regressor (after some parameter tuning)
lgbm = LGBMRegressor(objective = 'regression',
                     num_leaves = 20,
                     learning_rate = 0.03,
                     n_estimators = 200,
                     max_bin = 50,
                     bagging_fraction = 0.85,
                     bagging_freq = 4,
                     bagging_seed = 6,
                     feature_fraction = 0.2,
                     feature_fraction_seed = 7,
                     verbose = -1)

mse_cv_lgbm, r2_cv_lgbm = alg_fit_cv(lgbm, x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_lgbm*-1), 4))
print("R-squared: %s" % r2_cv_lgbm)


# ### XGB-Regressor

# In[51]:


# XGB-Regressor (after some parameter tuning)
xgb = XGBRegressor(n_estimators = 300,
                   max_depth = 2,
                   min_child_weight = 0,
                   gamma = 8,
                   subsample = 0.6,
                   colsample_bytree = 0.9,
                   objective = 'reg:squarederror',
                   nthread = -1,
                   scale_pos_weight = 1,
                   seed = 27,
                   learning_rate = 0.02,
                   reg_alpha = 0.006)

mse_cv_xgb, r2_cv_xgb = alg_fit_cv(xgb, x, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_xgb*-1), 4))
print("R-squared: %s" % r2_cv_xgb)


# ### Feature importance

# XGB-Regressor를 사용하여 가장 높은 정확도 점수를 얻었기 때문에 이를 이용한 모델로 Feature importance를 측정한다.

# In[52]:


# Model
mod = xgb.fit(x, y)

# Feature importance
df_feature_importance = pd.DataFrame(xgb.feature_importances_, index = x.columns, 
                                     columns = ['feature importance']).sort_values('feature importance', ascending = False)
df_feature_importance


# ### New features

# 높은 중요도를 가진 특징을 사용한다. 다만 일부 특징의 경우, 중복되거나 다른 특징에 가깝기 때문에 사용할 수 없다. 예를 들어, 경기당 필드골(FG 18/19)은 이미 경기당 포인트(PTS 18/19)에 포함되기 때문에 필요하지 않다.

# In[53]:


# Drop out features with low importance or which are redundant
x_new = x.loc[:, ['Age 21/22', 'GS 21/22', 'FGA 21/22','3P 21/22', '2PA 21/22', 'FT 21/22',
    'DRB 21/22', 'AST 21/22', 'PTS 21/22', 'PER 21/22', 'USG% 21/22','OBPM 21/22', 'VORP 21/22']]


# ### Final Model
# 

# In[54]:


# XGB-Regressor (after some parameter tuning)
xgb_new = XGBRegressor(n_estimators = 270,
                       max_depth = 2,
                       min_child_weight = 0,
                       gamma = 18,
                       subsample = 0.7,
                       colsample_bytree = 0.9,
                       objective = 'reg:squarederror',
                       nthread = -1,
                       scale_pos_weight = 1,
                       seed = 27,
                       learning_rate = 0.023,
                       reg_alpha = 0.02)

mse_cv_xgb, r2_cv_xgb = alg_fit_cv(xgb_new, x_new, y, 'neg_mean_squared_error', 'r2')

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_cv_xgb*-1), 4))
print("R-squared: %s" % r2_cv_xgb)


# ### Let's see how it performs on some test data

# In[55]:


# Split data now with x_new
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.2, random_state = 0)

# Use function to fit algorithm
y_pred_xgb, mse_xgb, r2_xgb, df_acc_xgb = alg_fit(xgb_new, x_train, y_train, x_test, 'Player', 'Salary 22/23', 
                                                  df1, metrics.mean_squared_error, metrics.r2_score)

print("Root Mean Squared Error: %s" % round(np.sqrt(mse_xgb), 4))
print("R-squared: %s" % r2_xgb)
df_acc_xgb.head(10)


# ### Visualization

# 예상 연봉과 실제 연봉 비교한다.

# In[56]:


df_acc_xgb.head(20).plot(x='Player', y=['Salary 22/23', 'y_pred'], kind="bar")


# In[ ]:




