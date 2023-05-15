# Project
### 오일 주입 스크류 압축기의 시계열 진동 데이터를 활용한 고장 유형 예측 모델 개발, Data Challenge at PHMAP 2021

# Intro 
### 🗓️ Date 
Project term : 2022.05.02 ~ 2022.06.12 </br>
Presentation Date : 2022.06.13 </br>

# Data Set 
### ✅ Source 
- Data Challenge at PHM Asia Pacific 2021, Data is provided by SK telecom <br/>
https://www.kaggle.com/competitions/phmap21-classification-task/data <br/>
csv file : <br/>
train_1st_Bearing.csv <br/>
train_1st_Looseness.csv <br/>
train_1st_Normal.csv <br/>
train_1st_Unbalance.csv <br/>
train_1st_high.csv <br/>
train_2nd_Bearing.csv <br/>
train_2nd_Looseness.csv <br/>
train_2nd_Unbalance.csv <br/>
train_3rd_Normal.csv <br/>
train_3rd_Unbalance.csv <br/>


### ✅ Characteristics of the dataset 
  * 특징: 오일 주입 스크류 압축기의 고장 유형마다의 진동값을 시계열 센싱 데이터로 수집하여 진동 데이터로 고장 유형을 예측하는 모델을 개발하는 것을 목표로함
  * 타입 : Time series data, Numerical Vibration Data
  * 피쳐 : 2개 (Motor Vibration, Screw Vibration)
  * 클래스 : Normal(정상), Bearing(베어링 결함), Unbalance(불균형), Looseness(V-벨트 풀림), high(V-벨트 크게 풀림) 총 5개 클래스가 존재하고 클래스마다 2, 3번씩 측정하였음

# Contents
- 데이터 불러오기
 - 양이 아주 많기때문에 chunck size를 한 후 묶음으로 불러옴
- 데이터 전처리
 - 시계열 진동 데이터 fft로 변환
 - fft 스펙트로그램으로 변환 한 후 클래스마다 차이점 확인
 - 데이터 사이즈가 아주 크므로 fft한 결과 중 클래스마다 중요한 주파수 대역을 가지는 index값을 추출한 후 나머지는 삭제
 - 각 클래스마다 중요한 주파수 대역대의 벡터를 다 결합하여 그 대역대만을 가지고 분류하는 것임
 - 각 클래스마다 라벨 값을 벡터에 추가하고 인덱스값을 기준으로 데이터셋마다 추출
- 모델링
 - 클래스마다의 특징 차이가 명확한 데이터셋의 단순 분류 문제이므로 Random Forest를 사용

### ✅ Best Model & Score
RandomForest </br>
  * 첫번째 시도 F1socre : 0.9802
  * 두번째 시도 F1socre : 0.9913
  * 세번째 시도 F1socre : 0.9789
