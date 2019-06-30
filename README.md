# HENVEN_PathPlanning

헤븐 경로팀을 위한 새로운 github repository

1.Lane Detection
  K-city에서 찍은 사진들 Lane_image 폴더에 넣어놓았습니다. 파란색 차선, 하얀 차선등을 detection하는 코드를 threshold를 사용해서 구현해봅시다
  lanedetectionexample에 시뮬레이터로 작성된 예제파일이 있습니다.
  (lane_detection_tutorial.py과 lane_util.py 파일만 보면 됨)
  
2.Lidar_Team
  라이더를 통해 데이터를 어떻게 받아오는지 client와 server 예제를 통해 확인해보고 이를 분리해서 cv2 나 matplot를 이용하여 그래프를 도시하고 장애물을   어떤 방식으로 파악하고 데이터를 어떻게 슬라이싱 할 지 공부해봅시다.
  
3. YOLO
  Making 파일을 통해서 학습시킬 Target 이미지를 background에 합성을 시켜 보고 여러가지 dataset을 만들어 봅시다.
  (조도,크기,회전등을 고려하여 dataset을 증폭)
