import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'N:\[03] 단기 작업\동아리 7월 pj 산림\신규주제 분석')  # 경로 세팅
plt.rc('font', family='Malgun Gothic')     # 글꼴 세팅

df = pd.read_csv(r'2022년\카드데이터\제천시 구매내역 데이터.csv', encoding='UTF-8', sep=',' )
data = df.copy()

# 열이 많아서 전부 보기위해서 세팅.
cols = data.columns
pd.set_option('display.max_seq_items', None); cols


#%%
### 취급액에 가장 영향을 주는 요소.
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor, plot_importance

v1_label = LabelEncoder()
v2_label = LabelEncoder()
v3_label = LabelEncoder()
gb3_label = LabelEncoder()
gb2_label = LabelEncoder()
sex_ccd_label = LabelEncoder()
daw_ccd_r_label = LabelEncoder()
apv_ts_label = LabelEncoder()

data.v1 = v1_label.fit_transform(data.v1)
data.v2 = v2_label.fit_transform(data.v2)
data.v3 = v3_label.fit_transform(data.v3)
data.gb3 = gb3_label.fit_transform(data.gb3)
data.gb2 = gb2_label.fit_transform(data.gb2)
data.sex_ccd = sex_ccd_label.fit_transform(data.sex_ccd)
data.daw_ccd_r = daw_ccd_r_label.fit_transform(data.daw_ccd_r)
data.apv_ts_dl_tm_r = apv_ts_label.fit_transform(data.apv_ts_dl_tm_r)

# 랜덤포레스트 중요변수
# rf_model = RandomForestRegressor()
# rf_model.fit(data.drop(['vlm'], axis = 1), data.vlm)
# sns.barplot(cols.drop(['vlm']), rf_model.feature_importances_)

# XGB 중요변수. > rf는 많이 느리니까 애 쓰자.
xgb_model = XGBRegressor()
xgb_model.fit(data.drop(['vlm'], axis = 1), data.vlm)

plot_importance(xgb_model)

# usec=이용건수 / ta_ym = 이용연월 / gb2 = 업종 소분류 와 유사한 cln_age_r = 연령
# v1 = 이용회원 거주지 / 성별 / 소비시간 / 업종 대분류 순이다.
# 즉 매출에 영향을 미치는건 구매횟수로 절대적 고객수가 중요하고, 관광시기, 업종과 연령대 순서로 중요하다고 할 수 있겠다.

#%%
# 여행에서 소비에 가장 중요한 영향을 미치는 요소 분석.

df2 =  pd.read_csv(r'2022년\카드데이터\NATIVE(2018.1_2022.4).csv', encoding='UTF-8', sep='|' )
df2 = df2[df2.gb3 == '여행']
data = df2.copy()

v1_label = LabelEncoder()
v2_label = LabelEncoder()
v3_label = LabelEncoder()
gb3_label = LabelEncoder()
gb2_label = LabelEncoder()
sex_ccd_label = LabelEncoder()
daw_ccd_r_label = LabelEncoder()
apv_ts_label = LabelEncoder()

data.v1 = v1_label.fit_transform(data.v1)
data.v2 = v2_label.fit_transform(data.v2)
data.v3 = v3_label.fit_transform(data.v3)
data.gb3 = gb3_label.fit_transform(data.gb3)
data.gb2 = gb2_label.fit_transform(data.gb2)
data.sex_ccd = sex_ccd_label.fit_transform(data.sex_ccd)
data.daw_ccd_r = daw_ccd_r_label.fit_transform(data.daw_ccd_r)
data.apv_ts_dl_tm_r = apv_ts_label.fit_transform(data.apv_ts_dl_tm_r)
data = data.astype({'cln_age_r':'int'})
# 랜덤포레스트 중요변수
# rf_model = RandomForestRegressor()
# rf_model.fit(data.drop(['vlm'], axis = 1), data.vlm)
# sns.barplot(cols.drop(['vlm']), rf_model.feature_importances_)

# XGB 중요변수. > rf는 많이 느리니까 애 쓰자.
xgb_model = XGBRegressor()
xgb_model.fit(data.drop(['vlm'], axis = 1), data.vlm)

plot_importance(xgb_model)

from sklearn.metrics import r2_score
r2s = r2_score(data.vlm, xgb_model.predict(data.drop(['vlm'], axis = 1)))
print(f'결정계수 : {r2s}')
# 결정계수 : 0.7538191375660448

#%%
# 위의 인사이트로 군집화를 해보자.
'''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.countplot(df.v1)
# 먼저 충북권의 제천 소비가 가장 많고, 그 외 소비가 평균적으로 구성되고 있다.
# 따라서 시외(관외) 소비에 대한 활성화 방안을 찾기위해 여행객을 대상으로 분석을 진행하였다.

inner_spender = df[df['v1'] == '충북']
outer_spender = df[df['v1'] != '충북']

# MUST CHECK
v1_label.classes_                # 원본 라벨 확인
v1_label.inverse_transform([16]) # 16번이 충북임을 확인

inner_spender_lab = data[data['v1'] == 16]
outer_spender_lab = data[data['v1'] != 16]

# K-mean 처리전 표준화.
kmean_scaler = StandardScaler()
treat_outer_spender_lab = kmean_scaler.fit_transform(outer_spender_lab)

kmean_model = KMeans()
kmean_model.fit(treat_outer_spender_lab)
pd.DataFrame(kmean_model.predict(outer_spender_lab)).value_counts()

# k 개수 선정
# 화면(figure) 생성
plt.figure(figsize = (10, 6))
for i in range(1, 7):
    # 클러스터 생성
    estimator = KMeans(n_clusters = i)
    ids = estimator.fit_predict(treat_outer_spender_lab)
    # 2행 3열을 가진 서브플롯 추가 (인덱스 = i)
    plt.subplot(3, 2, i)
    plt.tight_layout()
    # 서브플롯의 라벨링
    plt.title("K value = {}".format(i))
    plt.xlabel('ItemsBought')
    plt.ylabel('ItemsReturned')
    # 클러스터링 그리기
    plt.scatter(treat_outer_spender_lab[:,1], treat_outer_spender_lab[:,2], c=ids)  
plt.show()

# 내가 군집화(다변량에 대해 넘 모르고 있는것 같은데... )
'''
#%%

# kmaen은 안됨
# kmode / kproto 할것.
# https://antonsruberts.github.io/kproto-audience/ 검색해보면 애말고 결과 엄청엄청많음.


# Import module for data manipulation
import os
import pandas as pd
import numpy as np
from plotnine import *
import plotnine
os.chdir(r'E:\[03] 단기 작업\동아리 7월 pj 산림\신규주제 분석')  # 경로 세팅

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r'2022년\카드데이터\21년 여행 카테고리 소비내역.csv', encoding='UTF-8', sep=',' )
df['v2v3'] = df["v2"] + "-" + df["v3"]
df = df.drop(labels=['v2','v3'],axis=1)

# Inspect the data type
df.info()

# Inspect the categorical variables
df.select_dtypes('object').nunique()
# v2v3변수를 제외하곤 크게 범주가 많지 않음. 축소 필요 없어보임.
# 변수의 의미는 다 있는것으로 파악됨.-> apv_ts.... 변수가 그나마 가장 필요없을듯.

# Check missing value
df.isna().sum()

# fill na
df.v1 = df.v1.fillna('구분없음')

# Check missing value
df.isna().sum()

# The distribution of sales each region
df_region = pd.DataFrame(df['v2v3'].value_counts()).reset_index()
df_region['v2v3'] = df_region['v2v3'] / df['v2v3'].value_counts().sum()
df_region.rename(columns = {'index':'v2v3', 'v2v3':'Total'}, inplace = True)
df_region = df_region.sort_values('Total', ascending = True).reset_index(drop = True)
# The dataframe
df_region = df.groupby('v2v3').agg({
    'v2v3' : 'count',
    'v1' : lambda x:x.mode(), # 최빈값
    'gb3': lambda x:x.mode(),
    'gb2': lambda x:x.mode(),
    'sex_ccd': 'count',
    'cln_age_r': 'mean',
    'apv_ts_dl_tm_r': 'count',
    'vlm': 'mean',
    'usec': 'mean'
    }
).rename(columns = {'v2v3': 'Total'}).reset_index().sort_values('Total', ascending = True)

# 순수한 여행 요약통계량이 나오지.

# Data viz
'''
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = df_region)+
    geom_bar(aes(x = 'v3',
                 y = 'Total'))+
    geom_text(aes(x = 'v3',
                   y = 'Total',
                   label = 'Total'),
               size = 10,
               nudge_y = 120)+
    labs(title = 'Region that has the highest purchases')+
    xlab('지역')+
    ylab('Frequency')+
    scale_x_discrete(limits = df_region['v3'].tolist())+
    theme_minimal()+
    coord_flip()
)
일단 패스. 중요한거 아님.
'''


# Order the index of cross tabulation
order_region = df_region['v2v3'].to_list()
order_region.append('All')
# distribution of item type
df_item = pd.crosstab(df['v2v3'], df['v1'], margins = True).reindex(order_region, axis = 0).reset_index()
# Remove index name
df_item.columns.name = None
df_item
# 피벗 테입르 그려서 범주별 데이터 충분한지.

# Data pre-processing
# 열 차원 충분함. 건들필요 없음

# Show the data after pre-processing
print('Dimension data: {} rows and {} columns'.format(len(df), len(df.columns)))
df.head()


# obj-> categori 변환
# Get the position of categorical columns
df.month = df.month.astype('object') # 월은 일관성을 위해 obj 처리후 움직임
catColumnsPos = [df.columns.get_loc(col) for col in list(df.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(df.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))

df.info()

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

# Convert dataframe to matrix
dfMatrix = Scaler.fit_transform(df)
dfMatrix.shape
# 이거 변환하면서 열번호 바껴서 그럼.

#%%
# 엘보 포인트 구하기
########### 중요. 이거 실행시간 엄청 오래걸림 #############3
# i5-7300hq기준, 1iter 30분, 2iter 2시간반 필요, 3iter 3시간 소요. iter4 3시간 소요.
# 5600x 기준, iter5 5529.31s/it, 5073.15s/it

# cost = [7623249.499999979, 6197962.257189513, 5118320.627155959, 4524735.649103663, 4291327.922189697, 4015226.168555466,  ,3757158.409232641]

from kmodes.kprototypes import KPrototypes
from tqdm import tqdm

# Choose optimal K using Elbow method
cost = []
for cluster in tqdm(range(1, 10)):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0, verbose = 1)
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break
# Converting the results into a dataframe and plotting them
df_cost = pd.DataFrame({'Cluster':range(1, 7), 'Cost':cost}) # 6이상 하는게 무의미하다 판단. 여기까지 진행함.

# 얻은 결과값
# cost = [7623249.499999979, 6197962.257189513, 5118320.627155959, 4524735.649103663, 4291327.922189697, 4015226.168555466]

# Data viz
plotnine.options.figure_size = (8, 4.8)
(
    ggplot(data = df_cost)+
    geom_line(aes(x = 'Cluster',
                  y = 'Cost'))+
    geom_point(aes(x = 'Cluster',
                   y = 'Cost'))+
    geom_label(aes(x = 'Cluster',
                   y = 'Cost',
                   label = 'Cluster'),
               size = 10,
               nudge_y = 1000) +
    labs(title = 'Optimal number of cluster with Elbow Method')+
    xlab('Number of Clusters k')+
    ylab('Cost')+
    theme_minimal()
)


# 여기서 일단 스톱하고 결과 진행 확인할것. elbow point check
# 아래 코드는 재훈련 시키는거니까 위 엘보포인트는 한번만 진행하면 되지.
#%%
#%%
#%%
#%%
#%%
# 위에서 나온 적절 K값이 들어가야함
# 코드는 여기 참조
# https://towardsdatascience.com/the-k-prototype-as-clustering-algorithm-for-mixed-data-type-categorical-and-numerical-fe7c50538ebb
# 돌리면서 확인했는데 GB3을 지우자. 얘 동일값이자나.


# Fit the cluster
kprototype = KPrototypes(n_jobs = -1, n_clusters = 4, init = 'Huang', random_state = 0)
kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

# Cluster centorid
kprototype.cluster_centroids_
# Check the iteration of the clusters created
kprototype.n_iter_
# Check the cost of the clusters created
kprototype.cost_

'''
kprototype.cluster_centroids_
Out[6]: 
array([['-0.6309249486355075', '-0.03743608081477476',
        '-0.03262808335397237', '경기', '여행', '숙박', 'M', 'WHITE', '활동',
        '10', '제주-제주시'],
       ['1.1400081290135502', '-0.050142574065763335',
        '-0.04914132002851004', '경기', '여행', '숙박', 'M', 'WHITE', '활동',
        '10', '제주-제주시'],
       ['-0.39398113633934007', '43.38742713517076', '51.25952539007545',
        '서울', '여행', '교통', 'F', 'WHITE', '활동', '11', '대전-동구'],
       ['-0.17856385238143285', '8.945777147463305', '7.606086707525889',
        '서울', '여행', '교통', 'M', 'WHITE', '활동', '11', '대전-동구']],
      dtype='<U32')

kprototype.n_iter_
Out[7]: 25

kprototype.n_iter_
Out[8]: 25

kprototype.cost_
Out[9]: 4524735.649103663
'''


# Add the cluster to the dataframe
df['Cluster Labels'] = kprototype.labels_

# save cluster
df.to_csv('2021_4group_K_prototype.csv')


# 요약통계량 보는 작업. 저장한 데이터는 다른곳에서 분석해보자.
'''
df['Segment'] = df['Cluster Labels'].map({0:'First', 1:'Second', 2:'Third'})
# Order the cluster
df['Segment'] = df['Segment'].astype('category')
df['Segment'] = df['Segment'].cat.reorder_categories(['First','Second','Third'])


# Cluster interpretation
df.rename(columns = {'Cluster Labels':'Total'}, inplace = True)
df.groupby('Segment').agg(
    {
        'Total':'count',
        'Region': lambda x: x.value_counts().index[0],
        'Item Type': lambda x: x.value_counts().index[0],
        'Sales Channel': lambda x: x.value_counts().index[0],
        'Order Priority': lambda x: x.value_counts().index[0],
        'Units Sold': 'mean',
        'Unit Price': 'mean',
        'Total Revenue': 'mean',
        'Total Cost': 'mean',
        'Total Profit': 'mean'
    }
).reset_index()
'''



#%%
# 4그룹인 너무 분할이 안된것 같아서 8한번 해봄.
# 수정해서 k = 8 i5-7300hq 돌린결과, 소요시간 8시간..앵간하면 16코어로 돌리는게 좋을듯.

# Fit the cluster
kprototype = KPrototypes(n_jobs = -1, n_clusters = 8, init = 'Huang', random_state = 0)
kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

# Cluster centorid
kprototype.cluster_centroids_
'''
array([['-0.14151252689000673', '4.198864264061899', '3.718870924477486',
        '서울', '여행', '교통', 'M', 'WHITE', '활동', '10', '대전-동구'],
       ['1.283206724136071', '-0.0666695866730225',
        '-0.0610881531689152', '경기', '여행', '숙박', 'F', 'RED', '활동', '10',
        '제주-제주시'],
       ['0.6121364060941582', '-0.05538942594758897',
        '-0.05812630432015921', '서울', '여행', '숙박', 'M', 'WHITE', '휴식',
        '1', '인천-중구'],
       ['0.1586061615385706', '-0.05267059894356001',
        '-0.05118577974651268', '인천', '여행', '숙박', 'F', 'RED', '활동', '11',
        '제주-제주시'],
       ['-0.4282022322211847', '45.970717687307456', '55.78767542454829',
        '서울', '여행', '교통', 'F', 'WHITE', '활동', '5', '대전-동구'],
       ['-0.2228379550710543', '17.509302041613836',
        '14.033398378780912', '서울', '여행', '교통', 'M', 'WHITE', '활동', '11',
        '대전-동구'],
       ['-1.116652668014643', '-0.057548181879157695',
        '-0.047157531310638746', '서울', '여행', '숙박', 'M', 'WHITE', '활동',
        '10', '제주-제주시'],
       ['-0.8848632612559713', '-0.060080296306414246',
        '-0.04670352081660686', '경기', '여행', '숙박', 'F', 'RED', '활동', '5',
        '제주-제주시']], dtype='<U32')
'''
'''
ct = np.array([['-0.14151252689000673', '4.198864264061899', '3.718870924477486',
        '서울', '여행', '교통', 'M', 'WHITE', '활동', '10', '대전-동구'],
       ['1.283206724136071', '-0.0666695866730225',
        '-0.0610881531689152', '경기', '여행', '숙박', 'F', 'RED', '활동', '10',
        '제주-제주시'],
       ['0.6121364060941582', '-0.05538942594758897',
        '-0.05812630432015921', '서울', '여행', '숙박', 'M', 'WHITE', '휴식',
        '1', '인천-중구'],
       ['0.1586061615385706', '-0.05267059894356001',
        '-0.05118577974651268', '인천', '여행', '숙박', 'F', 'RED', '활동', '11',
        '제주-제주시'],
       ['-0.4282022322211847', '45.970717687307456', '55.78767542454829',
        '서울', '여행', '교통', 'F', 'WHITE', '활동', '5', '대전-동구'],
       ['-0.2228379550710543', '17.509302041613836',
        '14.033398378780912', '서울', '여행', '교통', 'M', 'WHITE', '활동', '11',
        '대전-동구'],
       ['-1.116652668014643', '-0.057548181879157695',
        '-0.047157531310638746', '서울', '여행', '숙박', 'M', 'WHITE', '활동',
        '10', '제주-제주시'],
       ['-0.8848632612559713', '-0.060080296306414246',
        '-0.04670352081660686', '경기', '여행', '숙박', 'F', 'RED', '활동', '5',
        '제주-제주시']])

pdct = pd.DataFrame(ct)

pdct.to_csv('centroid.csv')
'''
# Check the iteration of the clusters created
kprototype.n_iter_
# kprototype.n_iter_
# Out[8]: 29

# Check the cost of the clusters created
kprototype.cost_
'''
kprototype.cost_
Out[9]: 3757158.409232641
'''

# Add the cluster to the dataframe
df['Cluster Labels'] = kprototype.labels_

# save cluster
df.to_csv('2021_8group_K_prototype.csv')

