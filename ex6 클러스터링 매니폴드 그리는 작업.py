import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from datetime import datetime 
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
import umap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats

from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

from lightgbm import LGBMClassifier
import shap
from sklearn.model_selection import cross_val_score


# 데이터 임포트
os.chdir(r'E:\[03] 단기 작업\동아리 7월 pj 문화 관광 데이터 분석대회\신규주제 분석\2022년\카드데이터')

df = pd.read_csv('21년 여행 카테고리 소비내역.csv')

# 데이터 동일 전처리
df['v2v3'] = df["v2"] + "-" + df["v3"]
df = df.drop(labels=['v2','v3'],axis=1)


df.v1 = df.v1.fillna('구분없음')

# scaling
# 표준화 -> kmean은 표준화에 큰 영향받음.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

std_scaler1 = StandardScaler()
std_scaler2 = StandardScaler()

# df 순서 유지시키려 이렇게 작업.
Scaler = ColumnTransformer([
    ('Non_scale1', 'passthrough', ['v1','gb3','gb2','sex_ccd']),
    ("std_scale1", std_scaler1, ['cln_age_r']),
    ('Non_scale2', 'passthrough', ['daw_ccd_r', 'apv_ts_dl_tm_r']),
    ("std_scale2", std_scaler2, ['vlm', 'usec']),
    ('Non_scale3', 'passthrough', ['month','v2v3'])
])


# obj-> categori 변환
# Get the position of categorical columns
df.month = df.month.astype('object') # 월은 일관성을 위해 obj 처리후 움직임
catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(df.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

# Convert dataframe to matrix
dfMatrix = Scaler.fit_transform(df)
dfMatrix.shape

ndf = pd.DataFrame(dfMatrix, columns = ['v1','gb3','gb2','sex_ccd','cln_age_r','daw_ccd_r',
                                        'apv_ts_dl_tm_r','vlm', 'usec','month','v2v3'])

ndf.cln_age_r = ndf.cln_age_r.astype('float64')
ndf.vlm = ndf.vlm.astype('float64')
ndf.usec = ndf.usec.astype('float64')

ndf.info()


# 매니폴드 그리기.
#### 노트북(i5-7300hq, 시작시간 AM 09)
'''
WARNING: spectral initialisation failed! The eigenvector solver
failed. This is likely due to too small an eigengap. Consider
adding some noise or jitter to your data.

Falling back to random initialisation!
'''
#Preprocessing numerical
numerical = ndf.select_dtypes(exclude='object')

for c in numerical.columns:
    pt = PowerTransformer()
    numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))
    
##preprocessing categorical
categorical = ndf.select_dtypes(include='object')
categorical = pd.get_dummies(categorical)

#Percentage of columns which are categorical is used as weight parameter in embeddings later
categorical_weight = len(ndf.select_dtypes(include='object').columns) / ndf.shape[1]

#Embedding numerical & categorical
fit1 = umap.UMAP(metric='l2').fit(numerical)
fit2 = umap.UMAP(metric='dice').fit(categorical)

categorical_weight


#Augmenting the numerical embedding with categorical
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components, 
                                                fit1._initial_alpha, fit1._a, fit1._b, 
                                                fit1.repulsion_strength, fit1.negative_sample_rate, 
                                                200, 'random', np.random, fit1.metric, 
                                                fit1._metric_kwds, False)

plt.figure(figsize=(20, 10))
plt.scatter(*embedding.T, s=2, cmap='Spectral', alpha=1.0)
plt.show()

#############################



# 클러스터된 애의 매니폴드
clusterd = pd.read_csv(r'E:\[03] 단기 작업\동아리 7월 pj 문화 관광 데이터 분석대회\신규주제 분석\2021_8group_K_prototype.csv')

fig, ax = plt.subplots()
fig.set_size_inches((20, 10))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], s=2, c=clusterd['Cluster Labels'], cmap='tab20b', alpha=1.0)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(num=15),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)