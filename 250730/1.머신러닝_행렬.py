from sklearn.metrics import confusion_matrix #혼동 행렬 생성
from sklearn.metrics import classification_report # 성능평가 결과 확인

# 안녕윤아 어떻게 바뀌었는지 함께 확인해볼까?
# data_traget = [1, 0, 1, 0, 1, 1, 0, 0, 2, 3, 2, 3, 3, 2] # 실제 정답 데이터
# model_pred =  [1, 1, 1, 0, 1, 0, 1 ,0, 2, 1, 0, 3, 3, 1] # 모델이 예측한 값
#
# result = confusion_matrix(data_traget,model_pred)
# print(result)
# print(classification_report(data_traget,model_pred))
# # t_true = [0,1,1,0,1]
# # y_pred = [0,1,1,1,1]
#
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # knn 분류 모델
import numpy as np
import matplotlib.pyplot as plt
# 도미 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 데이터
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length #길이 데이터 병합
weight = bream_weight + smelt_weight # 무게 데이터 병합

#1.데이터 셋 준비
fishdata = [[l,w] for l, w in zip(length,weight)] #묶어서, 다시 꺼내는 작업!!!
# print(fishedata) # 학습 시킬 데이터 준비

# plt.scatter(length,weight)
# plt.show() # 그래프보면 bream이랑 smelt 차이남

fish_data = np.array(fishdata)
print(fishdata)


fish_taget = [1]*35+ [0]*14 #지도학습이기 때문에 정답데이터 만들어야함!!!!
print(fish_taget)


# 2. 모델 준비
model = KNeighborsClassifier(n_neighbors=5) # n_neighoor =5 (k개의 최근접 이웃을 5개로 디폴트 생성)

# 3. 모델 학습
model.fit(fish_data,fish_taget) # x==> 모델에 입력할 학습데이터, y ==> 학습데이터에 대한 정답 데이터

# 4. 모델 성능 평가

acc= model.score(fish_data,fish_taget) # 입력데이터에 대한 예측과 정답값을 혼동행렬 비교해서 정확도 산출
print(acc)


# 5. 완전 새로운 데이터를 가지고 모델 예측(추론)
pred = model.predict([[50, 300],[10,11]]) #학습은 필요없어 ^^
print(pred)

plt.scatter(length,weight)
plt.scatter(30,600,marker="D")
plt.scatter(10,11,marker="^")
plt.show()