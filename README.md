https://www.youtube.com/watch?v=hECRUHlTTqg

가장 먼저 드론 이미지 데이터셋을 images폴더에 넣어둔다.
classification.py를 먼저 실행하고 autolabel.py를 실행한다.
실패한 이미지들은 따로 직접 라벨링한다.

classification.py는 
드론 구별 코드 이미지데이터셋 폴더에 드론이있는지 없는지 인식하고 
있으면 successdir 성공폴더로 이미지 이동
없으면 faildir 실패폴더로 이미지를 이동하여 분리한다.
구별한 전체 이미지들은 images폴더에 넣으면 된다.
실행 방법 예 : python classification.py --modeldir=model --imagedir=images --successdir=success --faildir=fail


autolabel.py는 
자동라벨링코드이다. 
먼저 classification.py를 통해 분리된 확실히 드론이라고 인식된 이미지들이 모여있는 success폴더에
있는 이미지들에 객체 라벨링을 한다.
스페이스를 누르면 진행된다. 스페이스를 누르면 자동라벨링이되고 이상한 곳에 라벨링을 했다던가 라벨크기가 너무 크면
d 를 누르면 된다. 하다 지치거나 중단하고 싶으면 q를 누르면 된다.
실행 코드는 python autolabel.py --modeldir=model --imagedir=success --successdir=successfolder --faildir=fail

https://blog.naver.com/khw11044/221926134482
