'''
#################### PATH PLAN TEAM ####################

## ABOUT
- 미션 번호에 따라 lane과 lidar를 적절히 혼합하여 target point를 반환해주는 코드

## INPUT & OUTPUT
- input: 미션 번호
- output: 제어에 넘겨줄 path의 list

'''

class Combine:  # 나중에 이 함수에 display 메소드도 추가해야 할듯..?
    def __init__(self, mission_number):  # 초기화
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

    def get_map(self): return self.map
    def get_local_target(self): return self.local_target

    ########## 각 상황에 맞게 Lidar, Lane_Detection 이용하여 함수 짜기 ##########
    ## 신호/비신호는 path를 짜는것에 있어서는 같을 것 같아 하나로 묶음
    ## 각 상황에 맞는 map과 local target값을 넣으면
    def _path_tracking(self):
    def _static_obstacle(self):
    def _dynamic_obstacle(self):
    def _cross_left(self):
    def _cross_right(self):
    def _cross_straight(self):
    def _parking(self):