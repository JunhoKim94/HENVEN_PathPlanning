# 시뮬레이터에서 영상 받는 법

시뮬레이터 실행하고 Simulator-record.py를 연 다음 S버튼 누르면 Record라는 이름의 cv2 창이 뜹니다. <br>
실시간으로 받고 있는 카메라 영상을 보여줍니다.<br> 
esc 버튼을 누르면 창이 닫히고 output이라는 이름의 영상 파일로 저장됩니다. <br><br>
현재 설정된 카메라 위치는 아래와 같습니다. <br>
```python
CAM0_PARMS = {"SOCKET_TYPE": 'JPG',
        "WIDTH": 640, # image width
        "HEIGHT": 480, # image height
        "FOV": 90, # Field of view
        "localIP": "127.0.0.1",
        "localPort": 1232,
        "Block_SIZE": int(65000),
        "X": 0.5, # meter
        "Y": 0,
        "Z": 1,
        "YAW": 0, # deg
        "PITCH": -10,
        "ROLL": 0}

CAM1_PARMS = {"SOCKET_TYPE": 'JPG',
        "WIDTH": 640, # image width
        "HEIGHT": 480, # image height
        "FOV": 90, # Field of view
        "localIP": "127.0.0.1",
        "localPort": 1234,
        "Block_SIZE": int(65000),
        "X": 0.5, # meter
        "Y": 0,
        "Z": 1.5,
        "YAW": 0, # deg
        "PITCH": -10,
        "ROLL": 0}
```

# 시뮬레이터에서 로그 받는 법

시뮬레이터를 킨 다음 Simulator-record.py 실행합니다. <br>
1차원 배열 형태로 현재 상황의 데이터가 실시간으로 계속 출력됩니다. <br>
[gps, platform, main_cam, sub_cam, lidar] <br> <br>


## 기초적인 시뮬레이터 조작법

Q 버튼을 누르면 수동조작모드 // 기어는 N이 기본값이므로 기어조절 후 조작하기 <br>
U 버튼을 누르면 GUI 노출 <br>
화살표 키로 조작가능 <br>
+,- 버튼로 기어 조절 (1은 전진, N은 중립, R은 후진)

https://www.dropbox.com/sh/aehpeb0hj5zb9sq/AAAikMP8EySX3nxmWIl0siuMa/%EB%A7%A4%EB%89%B4%EC%96%BC%20%EB%B0%8F%20%EC%B0%B8%EA%B3%A0%EC%9E%90%EB%A3%8C?dl=0&preview=1_%EB%A7%A4%EB%89%B4%EC%96%BC+-+00+%EA%B0%9C%EC%9A%94.pdf&subfolder_nav_tracking=1
