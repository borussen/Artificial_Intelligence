---------------------------------------------------------------------------------

## 실습자료 13장 개요

> Recurrent Neural Networks 구현 (13장_RNN.ipynb) <br>
> RNN 계열의 GRU, LSTM 모델 구현 (13장_GRU+LSTM.ipynb) <br>
> Keras Reuter dataset에 RNN 적용 (13장_RNNforRueterdataset.ipynb) <br>
---------------------------------------------------------------------------------

### Recurrent Neural Networks 구현


sin 함수를 이용해 데이터를 생성하고 시각화한다. <br>

keras의 SimpleRNN 함수를 이용하여 기본적인 RNN 구조의 모델을 생성하고 데이터를 통해 모델을 학습시킨다. <br>

해당 모델의 epoch수 조절을 통해 모델의 변화를 줄 수 있다. <br>
위의 언급된 모델의 변화과정을 model 학습의 history에 저장하고 시각화한다. <br><br><br>




### RNN 계열의 GRU, LSTM 모델 구현


sin 함수를 이용해 데이터를 생성하고 시각화한다. <br>

keras의 GRU, LSTM 함수를 이용하여 GRU, LSTM 구조의 모델을 생성하고 데이터를 통해 모델을 학습시킨다. <br>

해당 모델의 epoch수 조절을 통해 모델의 변화를 줄 수 있다. <br>
위의 언급된 모델의 변화과정을 model 학습의 history에 저장하고 시각화한다. <br>

RNN 모델과의 차이를 해당 시각화를 통해 확인한다. <br><br><br>




### Keras Reuter dataset에 RNN 적용


keras 모듈에서 제공하는 텍스트 데이터인 reuter 데이터 셋을 앞서 실습해본 RNN 구조의 모델을 이용해 Classifier를 만들어본다. <br>

데이터 내부의 정보를 확인하고 pad_sequence등의 전처리를 거쳐 데이터를 활용하게 된다. <br>

모델의 학습에 과적합방지를 위해 활용되는 EarlyStopping을 실습해본다. <br>
