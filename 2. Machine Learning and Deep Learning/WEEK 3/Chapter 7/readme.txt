--------------------------------------------------------------------------------------------

실습자료 7장 readme.txt 개요

> KNN 분류기를 이용한 MNIST 데이터 분류 (7장_MNIST_knn분류.ipynb)
> Ensemble 모형을 이용한 make_moons 및 iris 데이터 분류 (7장_makemoons_iris_ensemble.ipynb)

---------------------------------------------------------------------------------------------



<KNN 분류기를 이용한 MNIST 데이터 분류>

KNN 분류 모형의 학습을 위하여 MNIST 데이터를 불러온다.
이 때, train set과 test set의 수를 다양하게 구성할 수 있다.

Train set의 순서를 shuffling한 후 KNeighborsClassifier 함수를 이용하여 KNN 분류 모형을 학습시킨다.
학습에 사용될 weight function을 의미하는 'weights' 매개변수에 'distance' 외에도 'uniform'등의 함수를 실험해볼 수 있다.
또한, 'n_neighbors' 매개변수에 이웃의 갯수를 자유롭게 설정해보면서 모델을 학습시켜본다.

학습이 완료되면 test set에서 성능을 체크해본다.





<Ensemble 모형을 이용한 make_moons 및 iris 데이터 분류>

Ensemble 분류 모형의 학습을 위하여 scikit-learn의 make_moons 데이터를 불러와 시각화해본다.

먼저 Bagging ensemble을 실습해본다.

Decision Tree 분류 모형을 베이스 모형으로 하여 'n_estimators' 매개변수를 통해 몇 개의 Decision Tree 모형을 앙상블에 참여시킬지 결정한다.
이번 실습에서는 500개의 Decision Tree 모형을 학습시켜 앙상블하였으며, 'n_estimators' 매개변수 값을 다양하게 실험해본다.
또한, Decision Tree외에도 다양한 분류 모형을 베이스로 앙상블을 학습시킬 수 있다.

그 다음 앙상블 모델과 단일 분류 모형의 성능을 체크한다. (이번 실습에서는 앙상블 모형이 약 90.4%, 단일 모형이 85.6%로 확인되었다.)
또한, 시각적으로 어떻게 분류하는지 알아보기 위해 Decision boundary를 그리는 함수를 정의하여 각각의 Decision boundary를 그려본다.


Boosting ensemble로는 Adaboost알고리즘을 실습해본다.

Adaboost 분류기에서도 위와 마찬가지로 'n_estimators' 매개변수를 적절하게 조절해보면서 학습해본다.
Learning rate도 다양한 값을 실험해본다.

Bagging ensemble과 마찬가지로, 학습된 boosting ensemble 모델을 앞서 정의한 함수를 이용하여 decision boundary를 시각화 해본다.


마지막으로 배깅 앙상블 모델의 한 종류인 random forest를 실습해본다.
Decision Tree를 Bagging ensemble한 모형과 random forest 모형을 같은 조건에서 학습시켜 분류 정확도를 도출해본다. (비슷한 결과 출력)
또한 학습된 random forest로 iris 데이터의 각 변수의 중요도를 도출해본다. 
  