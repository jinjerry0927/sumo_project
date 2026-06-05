"""E2(laneAreaDetector) 기반 관측함수 — 카메라 대신 현장 검지기로 state 구성.
sumo_rl DefaultObservationFunction 과 동일한 phase one-hot + min_green 머리에,
차로별 [대기열, 밀도, 정지수, occupancy] 4피처를 검지기에서 읽어 붙인다(0~1 정규화).
검지기 ID 규약 'e2_'+차로ID (make_e2_detectors.py 와 동일)."""
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

VEH_LEN = 7.5  # 차량 길이(5) + 최소 간격(2.5) 근사

class E2ObservationFunction(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        # sumo_rl 은 self.ts.lanes 설정 전에 관측함수를 생성하므로 지연 계산한다.
        self.det_ids = None
        self._caps = None

    def _ensure_caps(self):
        if self.det_ids is None:
            self.det_ids = ["e2_" + lane for lane in self.ts.lanes]
        if self._caps is None:
            self._caps = {d: max(self.ts.sumo.lanearea.getLength(d) / VEH_LEN, 1.0)
                          for d in self.det_ids}

    def __call__(self) -> np.ndarray:
        ts = self.ts
        self._ensure_caps()
        la = ts.sumo.lanearea
        phase_id = [1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)]
        min_green = [0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1]
        queue, density, halting, occ = [], [], [], []
        for d in self.det_ids:
            cap = self._caps[d]
            queue.append(min(1.0, la.getJamLengthVehicle(d) / cap))
            density.append(min(1.0, la.getLastStepVehicleNumber(d) / cap))
            halting.append(min(1.0, la.getLastStepHaltingNumber(d) / cap))
            occ.append(la.getLastStepOccupancy(d) / 100.0)
        return np.array(phase_id + min_green + queue + density + halting + occ,
                        dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        n = self.ts.num_green_phases + 1 + 4 * len(self.ts.lanes)
        return spaces.Box(low=np.zeros(n, dtype=np.float32),
                          high=np.ones(n, dtype=np.float32))
