------------------------------------------------------------------

실습자료 6장 readme.txt 개요

> 의사 결정 나무 모델의 학습과 시각화 (6장_의사결정나무모델.ipynb)

------------------------------------------------------------------


<의사 결정 나무 모델의 학습과 시각화>


iris 데이터 셋의 두 가지 feature(sepal length, sepal width) 만으로 의사 결정 나무 모델을 학습하고 각 점들을 
예측한 결과를 토대로 2차원 상 모델의 decision boundary를 확인한다.

제공된 코드의 경우 파라미터를(random_state = 0,
		               criterion = 'gini'
		               splitter = 'best'[splitter는 지정안할 시 디폴트 값인 'best'로 지정된다.]
			   max_depth=5)
로 지정되어 있으나,

random_state나 max_depth의 숫자 조절 / criterion을 'entropy'로 변경 / splitter를 'random'으로 변경해가며 시각화를 통해 decision boundary의 변화를 확인한다.




