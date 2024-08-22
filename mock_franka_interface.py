import numpy as np
import time

from frankateach.utils import FrequencyTimer
from frankateach.network import ZMQKeypointPublisher
from frankateach.messages import FrankaState

timer = FrequencyTimer(100)
publisher = ZMQKeypointPublisher("localhost", 8900)

while True:
  timer.start_loop()
  state = FrankaState(
    pos = np.random.rand(3),
    quat = np.random.rand(4),
    gripper = 0,
    timestamp = time.time()
  )
  publisher.pub_keypoints(state, "state")
  timer.end_loop()

