---------------------------------------------------------------------------------

실습자료 1장 개요

> 파이썬 개발 환경 구축 방법
> 파이썬 개발 플랫폼(jupyter notebook)에서 테스트 실행 (1장_테스트코드.ipynb)

---------------------------------------------------------------------------------



<파이썬 개발 환경 구축>

파이썬 개발 환경을 구축하기 위해서는 가장 먼저 Anaconda python 프로그램을 다운로드 받아 설치해야 한다.
(다운로드 사이트: https://www.anaconda.com/download/)

파이썬을 실행시킬 컴퓨터의 운영체제 및 bit 수를 확인하여 그에 맞는 버전을 설치한다.
(본 강의의 실습을 위해서는 python 3.7버전을 권장한다.)

첨부된 '아나콘다파이썬_설치과정1.png', '아나콘다파이썬_설치과정2.png'에 안내된대로 설치하면 된다.
 
설치가 완료되었으면, 'anaconda prompt'를 찾아 실행시킨다. (Anaconda가 설치된 파일에 함께 있다.)
anaconda prompt가 실행되면, 폴더명과 함께 오른쪽에 코드를 작성하도록 커서가 깜빡인다.

(base) C:\Users\user>

여기서 (base)는 가상환경을 지칭하는 부분인데, 아직 가상환경을 지정하지 않았으므로 기본 환경인 base가 출력된다.

그 다음, 가상환경을 새로 만들어 개발에 필요한 환경을 구축해야한다.
다음의 문법대로 가상환경을 만들 수 있다.

'conda create –n 가상환경명칭 python=버전번호'

본 강의에서는 가상환경명칭을 'machinelearning'으로 통일하기로 한다.

(base) C:\Users\user>conda create –n machinelearning python=3.7

위의 코드가 아무 오류 없이 실행되었다면 가상환경은 생성된 것이다.
이제 가상환경을 통해 파이썬 개발을 하기 위한 플랫폼인 'jupyter notebook'을 설치한다.

(base) C:\Users\user>conda install jupyter notebook

여기까지 모두 실행되었다면, 파이썬 개발 환경 및 플랫폼을 모두 생성한 것이다.
이제 위에서 생성한 가상환경을 활성화시켜본다.

(base) C:\Users\user>activate machinelearning

(machinelearning) C:\Users\user>

'machinelearning'이라는 가상환경이 활성화되면 (base) 대신 (machinelearning)으로 변경이 되는 것을 확인할 수 있다.
이제 해당 가상환경에서 개발에 필요한 파이썬 라이브러리들을 다음과 같이 'conda install' 명령어로 설치한다.

(machinelearning) C:\Users\user>conda install matplotlib
(machinelearning) C:\Users\user>conda install pandas
(machinelearning) C:\Users\user>conda install numpy
(machinelearning) C:\Users\user>conda install keras
(machinelearning) C:\Users\user>conda install scipy
(machinelearning) C:\Users\user>conda install scikit-learn
(machinelearning) C:\Users\user>conda install nb_conda
(machinelearning) C:\Users\user>conda install jupyter
(machinelearning) C:\Users\user>conda install ipykernel

하나씩 설치할 때마다 설치 여부를 묻는 '[y/n]?'이라는 질문이 출력되면 모두 y를 누르고 엔터를 클릭하면 된다.

또한, 나중에 jupyter notebook 플랫폼에서 'machinelearning'이라는 가상환경으로 진입하기 위해서는 다음의 코드를 실행시켜 플랫폼과 가상환경을 연결해주어야 한다.
 
(machinelearning) C:\Users\user>python -m ipykernel install --user --name machinelearning --display-name “machinelearning"

이제 다음의 코드를 통해 실제 파이썬 개발이 이루어질 jupyter notebook 플랫폼으로 이동해본다. 

(machinelearning) C:\Users\user>jupyter notebook

jupyter notebook으로 진입하면 기존에 컴퓨터에 있는 폴더들이 보일 것이다.
코드를 저장하고 싶은 폴더로 진입한 후 우측에 'New' 버튼을 클릭하면 'notebook' 카테고리에 앞서 생성했던 가상환경인  'machinelearning'이 보일 것이다.
'machinelearning'을 클릭하면 해당 가상환경으로 진입하게 된다.



<파이썬 개발 플랫폼(jupyter notebook)에서 테스트 실행>

마지막으로, 앞서 jupyter notebook 플랫폼에서 진입한 'machinelearning' 가상환경에서 파이썬 라이브러리들이 잘 설치되었는지와 파이썬 코드가 잘 작동하는지 테스트해본다.

첨부된 '1장_테스트코드.ipynb'에 나와있는대로 타이핑한 후 'shift+enter'를 동시에 클릭하면 코드가 실행된다.
에러 메시지 없이 정상적으로 결과를 출력하는지 확인하면 된다.
