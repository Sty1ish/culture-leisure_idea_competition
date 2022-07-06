#############################
######## data import ########
#############################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r'N:\[03] 단기 작업\동아리 7월 pj 산림\신규주제 분석')  # 경로 세팅

plt.rc('font', family='Malgun Gothic')     # 글꼴 세팅
data = pd.read_csv('2014-2021_data.csv', encoding='cp949', index_col = 'ID')


# 열이 많아서 전부 보기위해서 세팅.
cols = data.columns
pd.set_option('display.max_seq_items', None); cols



# 재형-2. 여가종류의 만족도 변화도 기간에 따라 분석하고.
# - 한번이상, 계속 참여한, 등등... 종류에 대해 가장 많은것 위주로 파악. 

#%%
#############################
######### EDA PART  #########
#############################

# 년도 데이터 분포
data.year.value_counts()
sns.countplot(y="year", data=data)


#######################################################
## 우리나라에서 즐긴적이 가장 많은(잘나가는) 여가활동##
#######################################################


# 한번 이상 참가한 여가활동.
data.Q1_1.value_counts() # 1~91번까지 전부 했음? 안했음으로 구성됨.
data.Q1_91.value_counts()

#%%
# 이건 그냥 합친 시각화를 해서 원형 그래프로 제시하는게 젤 나아보인다.

def pi_plot(df, title):
    temp = pd.DataFrame({'변수명':[], '참여인원수':[]})
    for idx, var in enumerate(cols[1:92]):
        try:
            temp.loc[idx] = [var, df.loc[:,var].value_counts().참여]
        except:
            temp.loc[idx] = [var, 0]
    plot = plt.pie(temp.참여인원수, labels = temp.변수명, autopct = '%.1f%%')
    plt.title(title)
    
#%% 
# 년도별 참가한 숫자.
# 조금 결합할 필요가 있지 않을까?
for y in data.year.unique():
    if y == 2014:
        print('무시')
        continue
    pi_plot(data[data['year'] == y], str(y) + '년도별 참가한 여가활동 비율'); plt.show()

# 일단 그지같이 나옴.

#%%
#######################################################
################ 가장 좋아하는  여가활동###############
#######################################################

# 귀찮으니까 함수화.
def top20_plot(df_col, title):
    temp = df_col.value_counts().head(20)
    plot = sns.barplot(y=temp.index, x=temp)
    plot.set_title(title)
    
#%%
# 지금까지 가장 좋아하는 여가활동(1위) 상위 20개
top20_plot(data.Q2_1_1, '가장 좋아하는 여가활동 (1위)')
top20_plot(data.Q2_1_2, '가장 좋아하는 여가활동 (2위)')
top20_plot(data.Q2_1_3, '가장 좋아하는 여가활동 (3위)')
top20_plot(data.Q2_1_4, '가장 좋아하는 여가활동 (4위)')
top20_plot(data.Q2_1_5, '가장 좋아하는 여가활동 (5위)')

# 조금 인사이트가 부족한것 같지? 
# 년도별 가장 좋아하는 여가활동 꺼내보자.
for y in data.year.unique():
    top20_plot(data[data['year'] == y].Q2_1_1, str(y) + '년 가장 좋아하는 여가활동 (1위)'); plt.show()

for y in data.year.unique():
    top20_plot(data[data['year'] == y].Q2_1_2, str(y) + '년 가장 좋아하는 여가활동 (2위)'); plt.show()

for y in data.year.unique():
    top20_plot(data[data['year'] == y].Q2_1_3, str(y) + '년 가장 좋아하는 여가활동 (3위)'); plt.show()

for y in data.year.unique():
    top20_plot(data[data['year'] == y].Q2_1_4, str(y) + '년 가장 좋아하는 여가활동 (4위)'); plt.show()

for y in data.year.unique():
    top20_plot(data[data['year'] == y].Q2_1_5, str(y) + '년 가장 좋아하는 여가활동 (5위)'); plt.show()

# 년도가 y축이고, 순위가 x축이고, 각각의 실선이 요소인 그래프.
# 위에 탐지결과 생각하면 년도별 좋아하는 여가활동의 차이가 거의 없음을 알수 있다.
# 상위 N개의 순위변화 측정하면 되지 않을까? (비율로 봐도 될것같기도 하고... 순위차트로 봐도 될것같기도 하고.)

#%%
# 대충 함수화.
def prop_lineplot(df, title, rank):
    rank_id = {1:'Q2_1_1', 2:'Q2_1_2', 3:'Q2_1_3', 4:'Q2_1_4', 5:'Q2_1_5'}
    rank = rank_id[rank] 

    data2014 = pd.DataFrame(df[df['year'] == 2014].loc[:,rank].value_counts().head(10) / df[df['year'] == 2014].year.count())
    data2016 = pd.DataFrame(df[df['year'] == 2016].loc[:,rank].value_counts().head(10) / df[df['year'] == 2016].year.count())
    data2018 = pd.DataFrame(df[df['year'] == 2018].loc[:,rank].value_counts().head(10) / df[df['year'] == 2018].year.count())
    data2019 = pd.DataFrame(df[df['year'] == 2019].loc[:,rank].value_counts().head(10) / df[df['year'] == 2019].year.count())
    data2020 = pd.DataFrame(df[df['year'] == 2020].loc[:,rank].value_counts().head(10) / df[df['year'] == 2020].year.count())
    data2021 = pd.DataFrame(df[df['year'] == 2021].loc[:,rank].value_counts().head(10) / df[df['year'] == 2021].year.count())   
                
    data2014['year'] = 2014; data2016['year'] = 2016; data2018['year'] = 2018
    data2019['year'] = 2019; data2020['year'] = 2020; data2021['year'] = 2021
    
    temp = pd.concat([data2014, data2016, data2018, data2019, data2020, data2021], axis = 0)
    temp = temp.reset_index()
    temp = temp.rename(columns={'index':'name','Q2_1_1':'응답비율', 'Q2_1_2':'응답비율', 'Q2_1_3':'응답비율',
                                'Q2_1_4':'응답비율', 'Q2_1_5':'응답비율'})
    sns.set(rc = {'figure.figsize':(11,8)})
    plt.rc('font', family='Malgun Gothic')
    
    plot = sns.lineplot(x = temp.year, y = temp.응답비율, hue = temp.name, data = temp, palette='Set1')
    lines = plot.get_lines()
    plt.setp(lines[0], linewidth = 4); plt.setp(lines[1], linewidth = 4); plt.setp(lines[2], linewidth = 4)
    plt.setp(lines[3], linewidth = 4); plt.setp(lines[4], linewidth = 4); plt.setp(lines[5], linewidth = 4)
    plt.setp(lines[6], linewidth = 4); plt.setp(lines[7], linewidth = 4); plt.setp(lines[8], linewidth = 4)
    plot.set_title(title)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize = 9)
    plt.tight_layout()
    
#%%

# 1-5위별 라인플롯.
prop_lineplot(data, '가장 좋아하는 여가활동 1위 흐름분석 - 두꺼운선이 위에 있는 범주', 1); plt.show()
prop_lineplot(data, '가장 좋아하는 여가활동 2위 흐름분석 - 두꺼운선이 위에 있는 범주', 2); plt.show()
prop_lineplot(data, '가장 좋아하는 여가활동 3위 흐름분석 - 두꺼운선이 위에 있는 범주', 3); plt.show()
prop_lineplot(data, '가장 좋아하는 여가활동 4위 흐름분석 - 두꺼운선이 위에 있는 범주', 4); plt.show()
prop_lineplot(data, '가장 좋아하는 여가활동 5위 흐름분석 - 두꺼운선이 위에 있는 범주', 5); plt.show()

###############################
###############################
###############################
### 순위별 정보 시각화 필요함.
### 동반자, 시간 등등...
###############################
###############################
###############################



#%%
#######################################################
################ 가장 좋아하는  여가활동###############
#######################################################

# q2_2_... 변수시리즈 <-앞에 변수랑 묶어서 분석해야함.





# Q3 여가활동 목적.
def leisure_obj_plot(data, title):
    temp = data.Q3.value_counts()
    plot = sns.barplot(y=temp.index, x=temp)
    plot.set_title(title)

#%%
leisure_obj_plot(data, '2014-2021년의 여가활동 목적')

# 년도별로 봐야할까?
for y in data.year.unique():
    leisure_obj_plot(data[data['year'] == y], str(y) + '년 여가활동 목적'); plt.show()
# 차이가 없다. 건강-자기만족은 오가긴 한데, 3위권 내에선 차이 X

#%%

# Q5 - 반복참여 변수와, 반복참여 횟수. Q6 .
# data.Q5.value_counts()보면 반반인거 봐서, 반복적인거 시각화 따로 해야할듯.
def rep_plot(df, title):
    temp = pd.DataFrame(df[df['Q5'] == '예'].Q6.value_counts().head(30))
    temp = temp.reset_index()
    temp = temp.rename(columns={'index':'name', 'Q6':'응답인원수'})
    sns.set(rc = {'figure.figsize':(6,10)}); plt.rc('font', family='Malgun Gothic')
    plot = sns.barplot(y=temp.name, x=temp.응답인원수)
    plot.set_title(title)


#%%    
rep_plot(data, '반복 참여하는 활동 순위')

# 이거 년도별로 봐야할까?

# 년도별로 봐야할까?
for y in data.year.unique():
    try:
        rep_plot(data[data['year'] == y], str(y) + '년 반복 참여하는 활동 순위'); plt.show()
    except:
        pass # 2014년은 해당 데이터가 없으므로 에러가 뜬다.





#%%




# 밑에, 평일 - 주말 참여도 위한 범주대로 나눠서 가장 좋아하는 여가활동 그려보는것.
# 범주 분리.

diction = {}
for idx, val in enumerate(['Q11_1_A','Q11_1_B','Q11_1_C','Q11_1_D','Q11_1_E','Q11_1_F','Q11_1_G','Q11_1_H']):
    temp = pd.DataFrame(data.loc[:,val].value_counts())
    temp = temp.reset_index()
    temp = temp.rename(columns={'index':'name', 'Q11_1_A':'응답인원수','Q11_1_B':'응답인원수','Q11_1_C':'응답인원수',
                                'Q11_1_D':'응답인원수', 'Q11_1_E':'응답인원수', 'Q11_1_F':'응답인원수', 
                                'Q11_1_G':'응답인원수', 'Q11_1_H':'응답인원수'})
    name_ex = ['문화예술 관람','문화예술 참여','스포츠 관람','스포츠 참여','관광','취미오락','휴식','사회 및 기타']
    diction.update({i: name_ex[idx] for i in temp.name if i not in ['없음', '무응답']})

diction
# 오류가 두개정도 포함됨.
# '57':'사회 및 기타'의 은 응답 오류로 원래 범주로 변경한다.
# '59': '관광'으로 나온 위치인데, 관광은 나와서는 안됨. 59번은 게임임.

del diction['59']
del diction['57']

diction['게임(온라인/모바일/콘솔게임 등)'] = '취미오락'
diction['홈페이지/블로그 관리'] = '취미오락'

# 결측값 대체용
diction['무응답'] = '무응답'
diction['없음'] = '없음'

# 이상하게 추가 안된값들 추가(아래 코딩 에러로 확인됨)
# -> 잘보면 아래 코딩에서 약간 다르게 코딩되어 있음. 범주가 통합됨.

diction['독서/만화책(웹툰)보기(2016, 2018)'] = '취미오락'
diction['이성교제(데이트)/미팅/소개팅(2016, 2018)'] = '사회 및 기타'
diction['친구만남/동호회 모임(2016, 2018)'] = '사회 및 기타'










# 이제 위에 한 작업을 딕셔너리로 치환한 값으로 재출력해보자.


# 파이플롯은 변수구조가 달라서 따로 인코딩 해야한다. 함수 새로만들것
# 1-8 / 9-15 / 16-19 / 20-37 / 38-48 / 49-70 + 89 / 71-79 / 80-88 + 90-91 변수.
# ['문화예술 관람','문화예술 참여','스포츠 관람','스포츠 참여','관광','취미오락','휴식','사회 및 기타'] 분포.
#for y in data.year.unique():
#    pi_plot(data[data['year'] == y], str(y) + '년도별 참가한 여가활동 비율'); plt.show()

# 이 개념 활용하면 되겠지
#for i in zip('1'*7,['1','2','3','4','5','6','7']):
#    print(i)


q1_diction = {}

for idx, val in zip(['Q1_'+str(i) for i in range(1,9)],['문화예술 관람']*8):
    q1_diction[idx] = val
    
for idx, val in zip(['Q1_'+str(i) for i in range(9,16)],['문화예술 참여']*7):
    q1_diction[idx] = val
    
for idx, val in zip(['Q1_'+str(i) for i in range(16,20)],['스포츠 관람']*4):
    q1_diction[idx] = val

for idx, val in zip(['Q1_'+str(i) for i in range(20,38)],['스포츠 참여']*18):
    q1_diction[idx] = val

for idx, val in zip(['Q1_'+str(i) for i in range(38,49)],['관광']*11):
    q1_diction[idx] = val

for idx, val in zip(['Q1_'+str(i) for i in range(49,71)],['취미오락']*22):
    q1_diction[idx] = val

q1_diction['Q1_89'] = '관광'

for idx, val in zip(['Q1_'+str(i) for i in range(71,80)],['휴식']*9):
    q1_diction[idx] = val
    
for idx, val in zip(['Q1_'+str(i) for i in range(80,89)],['사회 및 기타']*9):
    q1_diction[idx] = val
    
q1_diction['Q1_90'] = '사회 및 기타'
q1_diction['Q1_91'] = '사회 및 기타'

'''
for i in range(1, 92):
    print(q1_diction['Q1_'+str(i)])
check
'''

def summary_dict(df):
    summary_dict = {}
    for idx, val in q1_diction.items():
        try:
            if val in summary_dict:
                summary_dict[val] += df.loc[:,idx].value_counts().참여
            else:
                summary_dict[val] = df.loc[:,idx].value_counts().참여
        except:
            pass
    return(summary_dict)


# 총
temp = summary_dict(data)
temp = pd.DataFrame({'종목':temp.keys(), '수':temp.values()})
plt.pie(temp.수, labels = temp.종목, autopct = '%.1f%%'); plt.title('실제 참여한 활동 비율 (총)'); plt.show()
# 근데 중복범주에 여러번 참여할수도 있는 경우의 오류의 가능성이 존재하는 그래프임. 범주의 절대적 수가 달라서 그럼.

## 2016
plt.rc('font', family='Malgun Gothic') 
temp = summary_dict(data[data.year == 2016])
temp = pd.DataFrame({'종목':temp.keys(), '수':temp.values()})
plt.pie(temp.수, labels = temp.종목, autopct = '%.1f%%'); plt.title('16년 실제 참여한 활동 비율'); plt.show()
## 2018
plt.rc('font', family='Malgun Gothic') 
temp = summary_dict(data[data.year == 2018])
temp = pd.DataFrame({'종목':temp.keys(), '수':temp.values()})
plt.pie(temp.수, labels = temp.종목, autopct = '%.1f%%'); plt.title('18년 실제 참여한 활동 비율'); plt.show()
## 2019
plt.rc('font', family='Malgun Gothic') 
temp = summary_dict(data[data.year == 2019])
temp = pd.DataFrame({'종목':temp.keys(), '수':temp.values()})
plt.pie(temp.수, labels = temp.종목, autopct = '%.1f%%'); plt.title('19년 실제 참여한 활동 비율'); plt.show()
## 2020
plt.rc('font', family='Malgun Gothic') 
temp = summary_dict(data[data.year == 2020])
temp = pd.DataFrame({'종목':temp.keys(), '수':temp.values()})
plt.pie(temp.수, labels = temp.종목, autopct = '%.1f%%'); plt.title('20년 실제 참여한 활동 비율'); plt.show()
## 2021
plt.rc('font', family='Malgun Gothic') 
temp = summary_dict(data[data.year == 2021])
temp = pd.DataFrame({'종목':temp.keys(), '수':temp.values()})
plt.pie(temp.수, labels = temp.종목, autopct = '%.1f%%'); plt.title('21년 실제 참여한 활동 비율'); plt.show()


#%%
# 범주형 / 년도별 가장 좋아하는 여가활동 년도별.

# 에러 원인파악.
finder1 = set(diction)
finder2 = set(data.Q2_1_2.dropna())

print(finder2 - finder1) # dropna()가 필요하겠네.
del finder1, finder2

# 범주형 / 년도별 가장 좋아하는 여가활동 흐름분석

ndata = data.copy()

# 앞에서도 이 NA로 오류발생. 결측값 치환하자.
ndata.Q2_1_1.isna().sum()
ndata.Q2_1_2.isna().sum()
ndata.Q2_1_3.isna().sum()
ndata.Q2_1_4.isna().sum()
ndata.Q2_1_5.isna().sum()

#ndata는 얘한테만 쓰고 안쓸꺼니까 무시하고 한번에 결측값 채우자.
ndata = ndata.fillna('무응답')


ndata.Q2_1_1 = ndata['Q2_1_1'].map(diction)
ndata.Q2_1_2 = ndata['Q2_1_2'].map(diction)
ndata.Q2_1_3 = ndata['Q2_1_3'].map(diction)
ndata.Q2_1_4 = ndata['Q2_1_4'].map(diction)
ndata.Q2_1_5 = ndata['Q2_1_5'].map(diction)

# iris_df.target.replace(dict(enumerate(target_names))) 이런식으로도 치환가능.

#%%

# 1-5위별 라인플롯.
prop_lineplot(ndata, '가장 좋아하는 여가활동 1위 흐름분석', 1); plt.show()
prop_lineplot(ndata, '가장 좋아하는 여가활동 2위 흐름분석', 2); plt.show()
prop_lineplot(ndata, '가장 좋아하는 여가활동 3위 흐름분석', 3); plt.show()
prop_lineplot(ndata, '가장 좋아하는 여가활동 4위 흐름분석', 4); plt.show()
prop_lineplot(ndata, '가장 좋아하는 여가활동 5위 흐름분석', 5); plt.show()

#%%


# 년도별로 막대 플롯으로 비교해보자.

for y in ['Q2_1_1','Q2_1_2','Q2_1_3','Q2_1_4','Q2_1_5']:
    v2014 = ndata[(ndata['year'] == 2014)].loc[:,y].dropna().value_counts()
    v2016 = ndata[(ndata['year'] == 2016)].loc[:,y].dropna().value_counts()
    v2018 = ndata[(ndata['year'] == 2018)].loc[:,y].dropna().value_counts()
    v2019 = ndata[(ndata['year'] == 2019)].loc[:,y].dropna().value_counts()
    v2020 = ndata[(ndata['year'] == 2020)].loc[:,y].dropna().value_counts()
    v2021 = ndata[(ndata['year'] == 2021)].loc[:,y].dropna().value_counts()
    
    v2014 = pd.DataFrame({'year':'2014', 'var': v2014})
    v2016 = pd.DataFrame({'year':'2016', 'var': v2016})
    v2018 = pd.DataFrame({'year':'2018', 'var': v2018})
    v2019 = pd.DataFrame({'year':'2019', 'var': v2019})
    v2020 = pd.DataFrame({'year':'2020', 'var': v2020})
    v2021 = pd.DataFrame({'year':'2021', 'var': v2021})
    
    temp = pd.concat([v2014, v2016, v2018, v2019, v2020, v2021], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'year', data = temp)
    
    idx = ['Q2_1_1','Q2_1_2','Q2_1_3','Q2_1_4','Q2_1_5']
    
    plot.set_title('년도별 좋아하는 여가활동 범주'+str(idx.index(y)+1) + '위'); plt.show()

del v2014, v2016, v2018, v2019, v2020, v2021, idx #변수제거.

'''
개별로 보고싶을때.

for y in data.year.unique():
    temps = pd.Series([diction[i] for i in list(data[data['year'] == y].Q2_1_1.dropna())])
    top20_plot(temps, str(y) + '년 범주별 가장 좋아하는 여가활동 (1위)'); plt.show()

for y in data.year.unique():
    temps = pd.Series([diction[i] for i in list(data[data['year'] == y].Q2_1_2.dropna())])
    top20_plot(temps, str(y) + '년 범주별 가장 좋아하는 여가활동 (2위)'); plt.show()
    
for y in data.year.unique():
    temps = pd.Series([diction[i] for i in list(data[data['year'] == y].Q2_1_3.dropna())])
    top20_plot(temps, str(y) + '년 범주별 가장 좋아하는 여가활동 (3위)'); plt.show()
    
for y in data.year.unique():
    temps = pd.Series([diction[i] for i in list(data[data['year'] == y].Q2_1_4.dropna())])
    top20_plot(temps, str(y) + '년 범주별 가장 좋아하는 여가활동 (4위)'); plt.show()
    
for y in data.year.unique():
    temps = pd.Series([diction[i] for i in list(data[data['year'] == y].Q2_1_5.dropna())])
    top20_plot(temps, str(y) + '년 범주별 가장 좋아하는 여가활동 (5위)'); plt.show()

'''










#%%


# 성별과 나이대별로 다르게 확인해보자.


# DM1 성별 / DM2 연령1-7 / DM5 기혼여부 - 1미혼 2 기혼 3이혼 기타.


# 성별변수는 2018년부터 측정되기 시작하였음.
for y in [2018,2019,2020,2021]:
    female = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '여성')].Q2_1_1.dropna())],name='female').value_counts()
    male = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '남성')].Q2_1_1.dropna())],name='male').value_counts()
    female = pd.DataFrame({'sex':'female', 'var':female})
    male = pd.DataFrame({'sex':'male', 'var':male})
    temp = pd.concat([male, female], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'sex', data = temp)
    plot.set_title(str(y) + '년 범주별 남성이 가장 좋아하는 여가활동 (1위)'); plt.show()
    
for y in [2018,2019,2020,2021]:
    female = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '여성')].Q2_1_2.dropna())],name='female').value_counts()
    male = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '남성')].Q2_1_2.dropna())],name='male').value_counts()
    female = pd.DataFrame({'sex':'female', 'var':female})
    male = pd.DataFrame({'sex':'male', 'var':male})
    temp = pd.concat([male, female], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'sex', data = temp)
    plot.set_title(str(y) + '년 범주별 남성이 가장 좋아하는 여가활동 (2위)'); plt.show()
    
for y in [2018,2019,2020,2021]:
    female = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '여성')].Q2_1_3.dropna())],name='female').value_counts()
    male = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '남성')].Q2_1_3.dropna())],name='male').value_counts()
    female = pd.DataFrame({'sex':'female', 'var':female})
    male = pd.DataFrame({'sex':'male', 'var':male})
    temp = pd.concat([male, female], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'sex', data = temp)
    plot.set_title(str(y) + '년 범주별 남성이 가장 좋아하는 여가활동 (3위)'); plt.show()

    
for y in [2018,2019,2020,2021]:
    female = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '여성')].Q2_1_4.dropna())],name='female').value_counts()
    male = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '남성')].Q2_1_4.dropna())],name='male').value_counts()
    female = pd.DataFrame({'sex':'female', 'var':female})
    male = pd.DataFrame({'sex':'male', 'var':male})
    temp = pd.concat([male, female], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'sex', data = temp)
    plot.set_title(str(y) + '년 범주별 남성이 가장 좋아하는 여가활동 (4위)'); plt.show()
    
    
for y in [2018,2019,2020,2021]:
    female = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '여성')].Q2_1_5.dropna())],name='female').value_counts()
    male = pd.Series([diction[i] for i in list(data[(data['year'] == y) & (data['DM1'] == '남성')].Q2_1_5.dropna())],name='male').value_counts()
    female = pd.DataFrame({'sex':'female', 'var':female})
    male = pd.DataFrame({'sex':'male', 'var':male})
    temp = pd.concat([male, female], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'sex', data = temp)
    plot.set_title(str(y) + '년 범주별 남성이 가장 좋아하는 여가활동 (5위)'); plt.show()

del female, male # 변수제거.


#%%

# 연령대별 체크. 마찬가지로 18,19,20,21년 데이터만 있다.
data[(data['DM2'] == '40대')].year.value_counts()


dm2_temp = data.DM2.dropna().unique()
dm2_temp.sort()
for y in dm2_temp:
    v2018 = ndata[(data['DM2'] == y) & (ndata['year'] == 2018)].Q2_1_1.dropna().value_counts()
    v2019 = ndata[(data['DM2'] == y) & (ndata['year'] == 2019)].Q2_1_1.dropna().value_counts()
    v2020 = ndata[(data['DM2'] == y) & (ndata['year'] == 2020)].Q2_1_1.dropna().value_counts()
    v2021 = ndata[(data['DM2'] == y) & (ndata['year'] == 2021)].Q2_1_1.dropna().value_counts()
    
    v2018 = pd.DataFrame({'year':'2018', 'var': v2018})
    v2019 = pd.DataFrame({'year':'2019', 'var': v2019})
    v2020 = pd.DataFrame({'year':'2020', 'var': v2020})
    v2021 = pd.DataFrame({'year':'2021', 'var': v2021})
    
    temp = pd.concat([v2018, v2019, v2020, v2021], axis = 0)
    plot = sns.barplot(y=temp.index, x='var', hue = 'year', data = temp)
        
    plot.set_title(str(y) + '의 년도별 가장 좋아하는 여가활동 범주.'); plt.show()

del dm2_temp, v2018, v2019, v2020, v2021

# 가장 좋아하는 여가활동과 만족도간의 scatter 그려볼까

'''
예제 
ndata.iloc[:,92:93]
sns.pairplot(iris_df, hue = 'target')
plt.show()
# 커널 histogram과 scatter 를 같이그리는 플롯 hue 쓰는게 큰 특징.
print("\n[페어 플롯을 통한 데이터 확인]\n")
'''



# 여가비용 변수 체크 Q9 라인.



# D8 가구 소득에 따른 취미 차이?
# D9 대도시/시골, D10 권역으로, 즐기는거에 취미차이가 있을까 인사이트? 가능한가?


# 층화변수는 Q39라인부터.
# 개인요소는 DM 12까지.
