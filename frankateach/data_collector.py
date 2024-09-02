from collections import defaultdict
from pathlib import Path
import pickle
import cv2
import time
import threading
import h5py
import numpy as np

from frankateach.network import (
    ZMQCameraSubscriber,
    create_response_socket,
)
from frankateach.sensors.reskin import ReskinSensorSubscriber
from frankateach.utils import notify_component_start


from frankateach.constants import (
    HOST,
    CAM_PORT,
    STATE_PORT,
    DEPTH_PORT_OFFSET,
    RESKIN_STREAM_PORT,
)


class DataCollector:
    def __init__(
        self,
        stop_flag,
        storage_path: str,
        demo_num: int,
        cams=[],  # cam serial numbers
        cam_config=None,  # cam config
        collect_img=False,
        collect_state=False,
        collect_depth=False,
        collect_reskin=False,
    ):
        self.image_subscribers = []
        self.depth_subscribers = []
        if collect_img:
            for cam_idx, _ in enumerate(cams):
                self.image_subscribers.append(
                    ZMQCameraSubscriber(HOST, CAM_PORT + cam_idx, "RGB")
                )

        if collect_depth:
            for cam_idx, _ in enumerate(cams):
                self.depth_subscribers.append(
                    ZMQCameraSubscriber(
                        HOST, CAM_PORT + DEPTH_PORT_OFFSET + cam_idx, "Depth"
                    )
                )
        if collect_state:
            self.state_socket = create_response_socket(HOST, STATE_PORT)

        if collect_reskin:
            self.reskin_subscriber = ReskinSensorSubscriber()

        # stop flag
        self.stop_flag = stop_flag

        # Create the storage directory
        self.storage_path = Path(storage_path) / f"demonstration_{demo_num}"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        print("Storage path: ", self.storage_path)

        self.run_event = threading.Event()
        self.run_event.set()
        self.threads = []

        for cam_idx, _ in enumerate(cams):
            if collect_img:
                self.threads.append(
                    threading.Thread(
                        target=self.save_rgb,
                        args=(cam_idx, cam_config),
                        daemon=True,
                    )
                )

            if collect_depth:
                self.threads.append(
                    threading.Thread(
                        target=self.save_depth,
                        args=(cam_idx, cam_config),
                        daemon=True,
                    )
                )

        if collect_state:
            self.threads.append(threading.Thread(target=self.save_states, daemon=True))

        if collect_reskin:
            self.threads.append(threading.Thread(target=self.save_reskin, daemon=True))

    def start(self):
        for thread in self.threads:
            thread.start()

        while not self.stop_flag.value:
            time.sleep(0.01)
        print("Stopping the data collection...")
        self.run_event.clear()
        for thread in self.threads:
            thread.join()

    def save_rgb(self, cam_idx, cam_config):
        notify_component_start(component_name="RGB Image Collector")

        filename = self.storage_path / f"cam_{cam_idx}_rgb_video.avi"
        metadata_filename = self.storage_path / f"cam_{cam_idx}_rgb_video.metadata"

        recorder = cv2.VideoWriter(
            str(filename),
            cv2.VideoWriter_fourcc(*"XVID"),
            cam_config.fps,
            (cam_config.width, cam_config.height),
        )

        timestamps = []
        metadata = dict(
            cam_idx=cam_idx,
            width=cam_config.width,
            height=cam_config.height,
            fps=cam_config.fps,
            filename=filename,
            record_start_time=time.time(),
        )
        num_image_frames = 0
        print("Starting to record images from port:", CAM_PORT+cam_idx)

        while self.run_event.is_set():
            rgb_image, timestamp = self.image_subscribers[cam_idx].recv_rgb_image()
            # Save the image
            recorder.write(rgb_image)
            timestamps.append(timestamp)
            num_image_frames += 1

        print("finished recording")

        record_end_time = time.time()
        metadata["record_end_time"] = record_end_time
        metadata["num_image_frames"] = num_image_frames
        metadata["timestamps"] = timestamps
        recorder.release()
        with open(metadata_filename, "wb") as f:
            pickle.dump(metadata, f)

        print("Saved RGB video to ", filename)
        self.image_subscribers[cam_idx].stop()

    def save_depth(self, cam_idx, cam_config):
        pass

    def save_states(self):
        notify_component_start(component_name="State Collector")

        filename = self.storage_path / "states.pkl"
        states = []

        while self.run_event.is_set():
            state = pickle.loads(self.state_socket.recv())
            self.state_socket.send(b"ok")
            states.append(state)

        with open(filename, "wb") as f:
            pickle.dump(states, f)

        print("Saved states to ", filename)
        self.state_socket.close()

    def save_reskin(self):
        notify_component_start(component_name="Reskin Collector")

        sensor_information = defaultdict(list)
        filename = self.storage_path / "reskin_sensor_values.h5"

        print("Starting to record Reskin frames from port:", RESKIN_STREAM_PORT)

        while self.run_event.is_set():
            reskin_state = self.reskin_subscriber.get_sensor_state()
            for attr in reskin_state.keys():
                sensor_information[attr].append(reskin_state[attr])

        print("Finished recording Reskin frames")

        # Writing to dataset
        print("Compressing Reskin sensor data...")
        with h5py.File(filename, "w") as hf:
            for key in sensor_information.keys():
                sensor_information[key] = np.array(
                    sensor_information[key],
                    dtype=np.float32 if key != "timestamp" else np.float64,
                )
                hf.create_dataset(
                    key,
                    data=sensor_information[key],
                    compression="gzip",
                    compression_opts=6,
                )

        print("Saved Reskin sensor data to ", filename)
        print(
            "ReSkin Data duration: ",
            sensor_information["timestamp"][-1] - sensor_information["timestamp"][0],
        )

        print(
            "Frequency of ReSkin savings: ",
            len(sensor_information["timestamp"])
            / (
                sensor_information["timestamp"][-1] - sensor_information["timestamp"][0]
            ),
        )
