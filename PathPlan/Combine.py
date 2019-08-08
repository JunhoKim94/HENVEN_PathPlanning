'''
#################### PATH PLAN TEAM ####################

## ABOUT
- 미션 번호에 따라 lane과 lidar를 적절히 혼합하여 target point를 반환해주는 코드

## INPUT & OUTPUT
- input: 미션 번호
- output: 제어에 넘겨줄 path의 list

'''


from Lidar import Lidar
from Lane_Detection import Lane_Detection


class Combine:  # 나중에 이 함수에 display 메소드도 추가해야 할듯..?
    ## 초기화
    def __init__(self, mission_number):
        _mission_number = mission_number
        map = [(0,0)]
        local_target = (0,0)
        if self._mission_num == 0: self._path_tracking()
        elif self._mission_num == 1: self._static_obstacle()
        elif self._mission_num == 2: self._dynamic_obstacle()
        elif self._mission_num == 3: self._cross_straight()
        elif self._mission_num == 4: self._cross_left()
        elif self._mission_num == 5: self._cross_right()
        elif self._mission_num == 6: self._cross_left()
        elif self._mission_num == 7: self._cross_straight()
        elif self._mission_num == 8: self._parking()

    ## Path Plan 함수에 map을 넘겨줌
    def get_map(self): return self.map

    ## Path Plan 함수에 target점을 넘겨줌
    def get_local_target(self): return self.local_target

    ########## 각 상황에 맞게 Lidar, Lane_Detection 이용하여 함수 짜기 ##########
    ## 신호/비신호는 path를 짜는것에 있어서는 같을 것 같아 하나로 묶음
    ## 각 상황에 맞는 map과 local target값을 넣으면

    ######## ONLY CAM ########
    def _path_tracking(self):  # 기본주행

    ######## ONLY GPS ######## >> Map을 안넘겨 줘도 될듯..?
    def _cross_left(self):  # 경로 gps로 target점 결정
    def _cross_right(self):  # 경로 gps로 target점 결정
    def _cross_straight(self):  # 경로 gps로 target점 결정

    ####### WITH LIDAR #######
    def _static_obstacle(self):
        # map에 라이다까지 합성해서 기본주행을 쓰면 될듯
    def _dynamic_obstacle(self):
        # 장애물을 발견하기 전까지는 기본주행
        # lidar data x값이 양수인 애들중 차선보다 X값이 왼쪽에 있고 두 거리가 Y m 이하면 정지
        # 장애물을 발견하면 멈추고, 장애물이 범위 밖으로 벗어나면 1초 뒤 패킷 보내줌
    def _parking(self):

    ##########################################################################