############ data import #############
library(arulesViz)
library(arules)
library(data.table)
library(dplyr)

data = fread('E:/[03] 단기 작업/동아리 7월 pj 산림/신규주제 분석/2022년/카드데이터/제천시 구매내역 데이터.csv', encoding='UTF-8')

data$v1 = factor(data$v1); data$v2 = factor(data$v2); data$v3 = factor(data$v3)
data$gb3 = factor(data$gb3); data$gb2 = factor(data$gb2); data$sex_ccd = factor(data$sex_ccd)
data$daw_ccd_r = factor(data$daw_ccd_r); data$apv_ts_dl_tm_r = factor(data$apv_ts_dl_tm_r)
data$apv_ts_dl_tm_r = factor(data$apv_ts_dl_tm_r)
data$cln_age_r = factor(data$cln_age_r)

##########################################


############ 외수 세팅. #############
# 외수에 대해 조사.
df = data %>% filter(v1 != '충북')

str(df)
df.type1 = df %>% select(c(-v2, -v3, -ta_ym, -vlm, -usec, -gb2)) # 동일한 v2, v3, 수치형 변수, gb2-gb3연관으로 제거.

rules <- apriori(df.type1)
inspect(head(rules)) # 진짜 잘 나오는(확실한) 연관성은 파악되지 않음. 세부적으로 봐야함.
##########################################


######### gb3 = 취미오락 연관분석 #########
library(ggplot2)
# 지지도와 신뢰도의 값을 지정하면 좀 더 유의미한 결과가 나온다.
# 분야별 지지 / 신뢰도 확인 (수치 변화의 유의)

adj.rule <- apriori(df.type1, parameter = list(support=0.1, confidence = 0.6), 
                       appearance = list(rhs=c('gb3=취미오락'), default='lhs'),
                       control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', shading="confidence", edge.color="red", control = list(type='items', alpha=0.5))
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님

# 색깔 개선을 해봤는데도 라인 색이 나아지질 않음.
plot(adj.rule.sorted, method='graph', shading="confidence", control = list(
  edges = ggraph::geom_edge_link( end_cap = ggraph::circle(4, "mm"),
                                  start_cap = ggraph::circle(4, "mm"),
                                  color = "black",
                                  label_colour = 'black',
                                  label_alpha = 1,
                                  arrow = arrow(length = unit(3, "mm"), angle = 20, type = "closed", end = 'last'),
                                  alpha = .5
  )
))
plot(adj.rule.sorted,method="grouped")  # 그룹별 향상도 기준으로 지지도, 신뢰도를 볼까.

# 표현 잘하게 스샷찍을거면 얘가 확실하게 낫다.
plot(adj.rule.sorted, method='graph', engine = 'interactive')

##########################################


######### gb3 = 스포츠활동 연관분석 ########

# 아래 셋 모두 특징이. 신뢰도(조건이 주어졌을때 나올 확률.)/지지도(전체 거래에서 해당 비중)가 너무 낮다.
adj.rule <- apriori(df.type1, parameter = list(support=0.02, confidence = 0.2), 
                    appearance = list(rhs=c('gb3=스포츠활동'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님
# 스포츠와 관련된 활동이라 하면 골고루 퍼저있음이 확인됨.




##########################################


######### gb3 = 여행 연관분석 ########
adj.rule <- apriori(df.type1, parameter = list(support=0.02, confidence = 0.1), 
                    appearance = list(rhs=c('gb3=여행'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님
# 여행에는 주말 평일 활동시간, 비활동시간, 연령대 차이가 없어보인다.


##########################################


######### gb3 = 문화예술활동 연관분석 ########
adj.rule <- apriori(df.type1, parameter = list(support=0.01, confidence = 0.1), 
                    appearance = list(rhs=c('gb3=문화예술활동'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님

# 문화예술로는 여행오지 않는것으로 예상된다.
#> table(df$gb3)
#문화예술활동   스포츠활동         여행     취미오락 
# 6366        29634        23734       114917 
# 실제로 데이터는 이렇다.


##########################################



# gb2에 대해서 분석할 경우 너무 범주가 많아 분석이 잘 되지 않았다.
# 앞선 인사이트에서 여행(등산은 20, 30대가 많았고), 단순 자연경관 여행은 전반적으로 많았다.
# 지금 인사이트에서도 연령대는 꽤 중요한 연관성이 있었음.
# 즉 연령대별로 연관분석을 실시해 보았음.
######### 연령 ########

df.type2 = df %>% select(c(-v2, -v3, -ta_ym, -vlm, -usec)) # 이번기준은 gb2 gb3안떼고 해보자.

adj.rule <- apriori(df.type2, parameter = list(support=0.05, confidence = 0.2), 
                    appearance = list(rhs=c('cln_age_r=20'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)

adj.rule <- apriori(df.type2, parameter = list(support=0.05, confidence = 0.2), 
                    appearance = list(rhs=c('cln_age_r=30'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)

adj.rule <- apriori(df.type2, parameter = list(support=0.05, confidence = 0.2), 
                    appearance = list(rhs=c('cln_age_r=40'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)

adj.rule <- apriori(df.type2, parameter = list(support=0.05, confidence = 0.2), 
                    appearance = list(rhs=c('cln_age_r=50'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)

adj.rule <- apriori(df.type2, parameter = list(support=0.05, confidence = 0.2), 
                    appearance = list(rhs=c('cln_age_r=60'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)

# 글쎄... 인사이트가 나왔나?
# 취미 오락 파트에 다양한 연관성을 가지고 소비중이며, 연령대가 구성되어 있음.
# 60대의 여행-소비 비중은 다른 연령대에 비해 규칙성이 적음.



##########################################



############ 내수 세팅. #############
# 내수에 대해 조사.
df.inner = data %>% filter(v1 == '충북')
df.inner.type1 = df.inner %>% select(c(-v1, -v2, -v3, -ta_ym, -vlm, -usec, -gb2)) # 동일한 v2, v3, 수치형 변수, gb2-gb3연관으로 제거.



##########################################


######### gb3 = 취미오락 연관분석 #########
library(ggplot2)
# 지지도와 신뢰도의 값을 지정하면 좀 더 유의미한 결과가 나온다.
# 분야별 지지 / 신뢰도 확인 (수치 변화의 유의)

adj.rule <- apriori(df.inner.type1, parameter = list(support=0.1, confidence = 0.4), 
                    appearance = list(rhs=c('gb3=취미오락'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', shading="confidence", edge.color="red", control = list(type='items', alpha=0.5))
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님

# 색깔 개선을 해봤는데도 라인 색이 나아지질 않음.
plot(adj.rule.sorted, method='graph', shading="confidence", control = list(
  edges = ggraph::geom_edge_link( end_cap = ggraph::circle(4, "mm"),
                                  start_cap = ggraph::circle(4, "mm"),
                                  color = "black",
                                  label_colour = 'black',
                                  label_alpha = 1,
                                  arrow = arrow(length = unit(3, "mm"), angle = 20, type = "closed", end = 'last'),
                                  alpha = .5
  )
))
plot(adj.rule.sorted,method="grouped")  # 그룹별 향상도 기준으로 지지도, 신뢰도를 볼까.


##########################################


######### gb3 = 스포츠활동 연관분석 ########

# 아래 셋 모두 특징이. 신뢰도(조건이 주어졌을때 나올 확률.)/지지도(전체 거래에서 해당 비중)가 너무 낮다.
adj.rule <- apriori(df.inner.type1, parameter = list(support=0.03, confidence = 0.2), 
                    appearance = list(rhs=c('gb3=스포츠활동'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님
# 스포츠와 관련된 활동이라 하면 골고루 퍼저있음이 확인됨. 신뢰도가 작기때문에 타겟 설정이 안됨.


##########################################


######### gb3 = 여행 연관분석 ########
adj.rule <- apriori(df.inner.type1, parameter = list(support=0.02, confidence = 0.1), 
                    appearance = list(rhs=c('gb3=여행'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님
# 여행의 경우 지지도도 낮고, 신뢰도도 너무 낮아 의미가 없어보임.


##########################################


######### gb3 = 문화예술활동 연관분석 ########
adj.rule <- apriori(df.inner.type1, parameter = list(support=0.01, confidence = 0.1), 
                    appearance = list(rhs=c('gb3=문화예술활동'), default='lhs'),
                    control=list(verbose=F))
adj.rule.sorted <- sort(adj.rule, by='lift')
inspect(adj.rule.sorted)
plot(adj.rule.sorted, method="scatterplot") # 지지도, 신뢰도, 향상도의 산점도
plot(adj.rule.sorted, method='graph', engine = "igraph", control = list(type='items', alpha=0.5)) 
# support : 지지도. # lift : (향상도=신뢰도/지지도). 1보다 크면 우연하게 나온 확률이 아님
# 


##########################################

# lhs를 가지고 있을때 rhs가 스포츠, 여행, 문화예술, 취미오락중에 하나를 결정할 확률이 연관분석.
# 전반적으로 어떤 선택지를 고르더라도, 스포츠 소비, 여행소비, 문화예술 소비를 고를 확률은 낮고, 취미오락을 결정하는 비율이 0.6 이상임.
# 외수에 입장에서는 소비 비중이 어떻게 구성되는지 파악되지.
# 근데 반대로 내수의 입장에서는 이게 의미가 없는게 4개 다 소비를 할거잖아. 소비 비중 외적인 의미에서 의미가 있나....

# 취미 오락의 비중은 이렇게 되지
df %>% filter(gb3 == '취미오락') %>% select(gb2) %>% table()
# 외식이 가장 많았고, 종합쇼핑이 둘째로 많았다. 이상하게 미용이 상위권을 차지했으며(충북 사람의 소비가 아닌데도?) 그다음 패션쇼핑이 차지했다.

# 그리고 위에 플롯. 표현 잘하게 스샷찍을거면 얘가 확실하게 낫다.
# plot(adj.rule.sorted, method='graph', engine = 'interactive')

##########################################
### 아래부분은 몰라서 패스.
##########################################


##########################################
############ 계층적 군집분석 #############
library(cluster)
library(stringr)
library(data.table)
library(dplyr)
# library(bit64) # 구매액 데이터가 int64가 필요함., fread문제.

#Euclidean : 두 점 사이의 거리를 구할 때 가장 많이 쓰는 방식으로, 식은 다음과 같습니다.
#Manhattan : 두 점 사이의 절대적 거리를 이용한 거리 계산 방식으로 다음과 같습니다.
#Maximum : 두 점 사이의 거리가 좌표 차원에서의 가장 큰 벡터공간에서 정의됩니다.
#Gower : 양적변수가 포함되어 있을때도 사용할 수 있는 방법으로, 우선 선택된 변수들을 [0,1]사이의 값으로 표준화 시킨 후, 모든 변수들간의 거리를 가중평균하여 합한 값을 사용합니다.
#Complete : 최장연결법으로, 두 군집간의 최장 거리를 군집간 거리로 정의합니다.
#Single : 최단연결법으로, 두 군집간의 최단 거리를 군집간 거리로 정의합니다.
#Ward.D : Ward가 제안한 방법으로, 군집간의 거리보다는 군집내의 편차제곱합에 근거를 두고 군집을 병합하는 방법입니다. 군집을 병합하는 과정에서 생기는 정보의 손실이 최소가 되도록 정의합니다.
#Ward.D2 : Ward.D 방법에 표준화 수치를 사용한 것으로 절대값 대신 거듭제곱값을 사용합니다.
#Average : 평균연결법으로 각 군집에 속한 모든 개체들간의 거리의 평균으로 정의합니다.
#Mcquitty : 산술평균을 이용한 가중 쌍그룹 방법 (Weighted Pair Group Method with Arithmetic Means; WPGMA) 으로, 가장 가까운 두 군집이 합쳐져 하나의 그룹을 형성한 후, 다른 군집과의 거리는 산술평균으로 구합니다.
#Median : 중앙연결법으로, 군집간의 거리를 군집의 모든 샘플의 중앙값으로 정의하는 것입니다.
#Centroid : 중심연결법으로, 두 군집간의 거리가 두 군집의 중심간 거리로 정의됩니다. 여기서, s,t 는 각 군집의 중심점을 나타냅니다.


setwd('E:/[03] 단기 작업/동아리 7월 pj 산림/신규주제 분석/2022년/카드데이터')
df.type3 = fread('NATIVE(2018.1_2022.4).csv', encoding = 'UTF-8')

df.type3$year = substr(df.type3$ta_ym,1,4)
df.type3$month = factor(substr(df.type3$ta_ym,5,6), levels = c('01','02','03','04','05','06','07','08','09','10','11','12'))

# 램 딸려서 파일로 저장하고 작업중단.
df.type.2021 = df.type3 %>% filter((df.type3$year == '2021') & (df.type3$gb3 == '여행')) %>% select(-c('ta_ym','year'))
fwrite(df.type.2021, '21년 여행 카테고리 소비내역.csv'); rm(df.type.2021)
df.type.2020 = df.type3 %>% filter((df.type3$year == '2020') & (df.type3$gb3 == '여행')) %>% select(-c('ta_ym','year'))
fwrite(df.type.2020, '20년 여행 카테고리 소비내역.csv'); rm(df.type.2020)
df.type.2019 = df.type3 %>% filter((df.type3$year == '2019') & (df.type3$gb3 == '여행')) %>% select(-c('ta_ym','year'))
fwrite(df.type.2019, '19년 여행 카테고리 소비내역.csv'); rm(df.type.2019)
df.type.2018 = df.type3 %>% filter((df.type3$year == '2018') & (df.type3$gb3 == '여행')) %>% select(-c('ta_ym','year'))
fwrite(df.type.2018, '18년 여행 카테고리 소비내역.csv'); rm(df.type.2018)
rm(df.type3)


# 램 문제때문에 아래 내역은 하나씩 불러오기 진행후 진행할것. 위 작업도 램 20기가 이상이어야 수월히 동작함.
# 파이썬으로 넘어가.



temp.df = fread('21년 여행 카테고리 소비내역.csv', encoding = 'UTF-8')

temp.df$v1 = factor(temp.df$v1); temp.df$v2 = factor(temp.df$v2); temp.df$v3 = factor(temp.df$v3)
temp.df$gb3 = factor(temp.df$gb3); temp.df$gb2 = factor(temp.df$gb2); temp.df$sex_ccd = factor(temp.df$sex_ccd)
temp.df$daw_ccd_r = factor(temp.df$daw_ccd_r); temp.df$apv_ts_dl_tm_r = factor(temp.df$apv_ts_dl_tm_r)
temp.df$apv_ts_dl_tm_r = factor(temp.df$apv_ts_dl_tm_r)
temp.df$cln_age_r = factor(temp.df$cln_age_r); temp.df$month = factor(temp.df$month);
temp.df$vlm = temp.df$vlm / 10000

temp.df2 = temp.df %>% select(c(-'gb3',-'vlm',-'usec')) # gb3은 여행으로 통일. 수치형은 x
#library(caret)
#dummyVars()


gower_distance <- daisy(head(temp.df2,1000), metric = c("gower"))
# 무친 개념이라 실행이 불가능하다. Error: cannot allocate vector of size 175.2 Gb 에러니 최소한 차원을 줄여야 할듯.
class(gower_distance)

divisive_clust <- diana(as.matrix(gower_distance), 
                        diss = TRUE, keep.diss = TRUE)
plot(divisive_clust, main = "Divisive")

# 여기에서는 일반적으로 사용되는 Complete Linkages 활용합니다. 
agg_clust_c <- hclust(gower_distance, method = "complete")
plot(agg_clust_c, main = "Agglomerative, complete linkages")




# https://rpubs.com/Evan_Jung/hierarchical_clustering / 범주형 클러스터링 알고리즘.
# 아니 그냥 이해못함/ ㅇㅁㅇ
# kmean 방법


# knode 방법을 활용해보자
# 이론 : https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/
# r 코드 : https://www.rdocumentation.org/packages/klaR/versions/0.6-15/topics/kmodes
# 파이썬 코드 : https://data-newbie.tistory.com/739

library(klaR)
## run algorithm on x:
# 그룹수 어케정함??
(cl <- kmodes(temp.df2, 2))

## and visualize with some jitter:
plot(jitter(temp.df2), col = cl$cluster)
points(cl$modes, col = 1:5, pch = 8)

# kproto
# 수치0범주 통합 https://www.rdocumentation.org/packages/clustMixType/versions/0.2-15/topics/kproto



##########################################


# 주석.
# https://rdrr.io/rforge/arulesViz/man/plot.html # 연관분석 함수 플롯 사용법

