---------------------------------------------------------------------------------

실습자료 12장 readme.txt 개요

> Convolutional Neural Networks (12장_CNNforMNIST.ipynb)

---------------------------------------------------------------------------------



<Convolutional Neural Networks>


합성곱 신경망 모델(CNN)을 python 패키지 'keras'를 통해 구현한다.

코드는 다음과 같이 진행된다.

1. mnist 데이터의 로드
2. 데이터 전처리
3. 모델 생성(선언)
4. 모델 학습
5. 결과 확인
6. CNN layer 추가 후 결과확인

해당 코드의 3. 모델 생성 파트에서 CNN layer와 pooling layer의 추가인
model2.add(Conv2D(32,(3,3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
에서 매개변수인 filter size (3,3)이나 activation function을 'sigmoid' 등으로 변경해 결과를 비교할 수 있다.


해당 코드의 model.fit() 함수의 parameter인 batch_size와 epochs를 조절하여 학습을 조절하고 결과를 토대로 적절한 매개변수를 찾을 수 있다.

6.의 CNN layer 추가 후 성능이나 적절한 매개변수의 차이를 통해 CNN layer 추가의 효과를 확인할 수 있다.