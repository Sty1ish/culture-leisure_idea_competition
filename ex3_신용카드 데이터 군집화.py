import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'E:\[03] 단기 작업\동아리 7월 pj 산림\신규주제 분석')  # 경로 세팅
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
