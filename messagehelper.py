import json
import datetime
from PIL import Image
from io import BytesIO
import base64
from dataclasses import dataclass
import os
import sys

import logging
import ssl
import requests

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
log.addHandler(handler)

DECIMALS_TO_ROUND = 3


@dataclass
class Flower:
    index: int
    class_name: str
    score: float
    width: int
    height: int

    def to_dict(self):
        return {
            "index": self.index,
            "class_name": self.class_name,
            "score": round(float(self.score), DECIMALS_TO_ROUND),
            "size": [self.width, self.height],
        }


@dataclass
class Pollinator:
    index: int
    flower_index: int
    class_name: str
    score: float
    width: int
    height: int
    crop: Image

    def to_dict(self, save_crop=True):

        pollintor_dict = {
            "index": self.index,
            "flower_index": self.flower_index,
            "class_name": self.class_name,
            "score": round(self.score, DECIMALS_TO_ROUND),
            "crop": None,
        }
        if save_crop:
            bio = BytesIO()
            self.crop.save(bio, format="JPEG")
            bio.seek(0)
            encoded_image = base64.b64encode(bio.read()).decode("utf-8")
            pollintor_dict["crop"] = encoded_image
        return pollintor_dict


class MessageParser:
    def __init__(self):
        self.node_id = None
        self.timestamp = None
        self.images = []
        self.classes = []
        self.scores = []
        self.inference_times = None
        self.capture_duration = None
        self.image_src = None
        self.original_image_size = None
        self.conf_threshold = None
        self.iou_threshold = None
        self.max_det = None
        self.num_detections = 0
        self.margin = None
        self.model_name = None
        self.msg = None

    def parse_message(self, msg):
        self._clear()
        try:
            if not type(msg) is dict:
                msg = json.loads(msg)
            self.msg = msg
            meta = msg["metadata"]
            self.node_id = meta["node_id"]
            self.timestamp = datetime.datetime.fromisoformat(meta["capture_timestamp"])
            original_image_meta = meta["original_image"]
            self.original_image_size = original_image_meta["size"]
            self.capture_duration = original_image_meta["capture_duration"]
            self.image_src = original_image_meta["source"]
            flower_inference_meta = meta["flower_inference"]

            self.conf_threshold = flower_inference_meta["confidence_threshold"]
            self.iou_threshold = flower_inference_meta["iou_threshold"]
            self.margin = flower_inference_meta["margin"]
            self.model_name = flower_inference_meta["model_name"]
            self.max_det = flower_inference_meta["max_det"]
            detections = msg["detections"]
            for i in range(len(detections["flowers"])):
                self.classes.append(detections["flowers"][i]["class_name"])
                self.scores.append(detections["flowers"][i]["score"])
                self.images.append(self._load_image(detections["flowers"][i]["crop"]))
            self.num_detections = len(detections["flowers"])
            return True
        except Exception as e:
            print(e)
            return False

    def _clear(self):
        self.node_id = None
        self.timestamp = None
        self.images = []
        self.classes = []
        self.scores = []
        self.inference_times = None
        self.capture_duration = None
        self.original_image_size = None
        self.conf_threshold = None
        self.iou_threshold = None
        self.num_detections = 0
        self.margin = None
        self.model_name = None

    def _load_image(self, img_b64):
        im = Image.open(BytesIO(base64.b64decode(img_b64)))
        return im

    def get_metadata(self):
        metadata = {}
        metadata["node_id"] = self.node_id
        metadata["capture_timestamp"] = self.timestamp.isoformat()
        metadata["original_image"] = {}
        metadata["original_image"]["size"] = self.original_image_size
        metadata["original_image"]["capture_duration"] = self.capture_duration
        metadata["original_image"]["source"] = self.image_src
        metadata["flower_inference"] = {}
        metadata["flower_inference"]["confidence_threshold"] = self.conf_threshold
        metadata["flower_inference"]["iou_threshold"] = self.iou_threshold
        metadata["flower_inference"]["margin"] = self.margin
        metadata["flower_inference"]["model_name"] = self.model_name
        metadata["flower_inference"]["max_det"] = self.max_det
        metadata["flower_inference"]["inference_times"] = self.inference_times
        return metadata

    def print_detections(self):
        print("Node ID: {}".format(self.node_id))
        print("Timestamp: {}".format(self.timestamp))
        print("Process time: {}".format(self.inference_times))
        print("Download time: {}".format(self.capture_duration))
        print("Original image size: {}".format(self.original_image_size))
        print("Confidence threshold: {}".format(self.conf_threshold))
        print("IoU threshold: {}".format(self.iou_threshold))
        print("Number of detections: {}".format(self.num_detections))
        for i in range(len(self.classes)):
            print(
                "#{}: Class: {}".format(i, self.classes[i])
                + " Score: {}".format(self.scores[i])
                + " Crop size: {}".format(self.images[i].size)
            )

    def store_message(self, path):
        with open(path, "w") as f:
            json.dump(self.msg, f)
        return True


class MessageGenerator:
    def __init__(self):
        self.node_id = None
        self.timestamp = None
        self.flowers = []
        self.pollinators = []
        self.metadata = {}

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def set_node_id(self, node_id):
        self.node_id = node_id

    def add_metadata(self, metadata: dict, key: str):
        self.metadata[key] = metadata

    def set_metadata(self, metadata: dict):
        self.metadata = metadata

    def add_flower(self, flower: Flower):
        self.flowers.append(flower)

    def add_pollinator(self, pollinator: Pollinator):
        self.pollinators.append(pollinator)

    def generate_message(self, save_crop=True):
        flowers = []
        pollinators = []
        for flower in self.flowers:
            flowers.append(flower.to_dict())
        for pollinator in self.pollinators:
            pollinators.append(pollinator.to_dict())
        flowers.sort(key=lambda x: x["index"])
        pollinators.sort(key=lambda x: x["index"])

        message = {
            "detections": {"flowers": flowers, "pollinators": pollinators},
            "metadata": self.metadata,
        }
        return message

    def generate_filename(self, format=".json"):
        filename = (
            self.node_id + "_" + self.timestamp.strftime("%Y-%m-%dT%H-%M-%SZ") + format
        )
        return filename

    def _generate_save_path(self):
        date_dir = self.timestamp.strftime("%Y-%m-%d")
        time_dir = self.timestamp.strftime("%H")
        return self.node_id + "/" + date_dir + "/" + time_dir + "/"

    def store_message(self, base_dir, save_crop=True):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not base_dir.endswith("/"):
            base_dir += "/"
        filepath = base_dir + self._generate_save_path()
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            log.info("Created directory: {}".format(filepath))
        with open(filepath + self.generate_filename(), "w") as f:
            json.dump(self.generate_message(save_crop=save_crop), f)
        log.info("Saved message to: {}".format(filepath + self.generate_filename()))
        return True


class MQTTClient:
    def __init__(self, host, port, topic, username, password, use_tls):
        self.host = host
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.use_tls = use_tls
        if self.username is not None and self.password is not None:
            self.auth = {
                "username": self.username,
                "password": self.password,
            }
        else:
            self.auth = None

    def publish(self, message, filename=None, node_id=None, hostname=None):
        import paho.mqtt.publish as publish

        topic = self.topic
        if filename is not None:
            topic = topic.replace("${filename}", filename)
        if node_id is not None:
            topic = topic.replace("${node_id}", node_id)
        if hostname is not None:
            topic = topic.replace("${hostname}", hostname)
        log.info("Publishing to {} on topic: {}".format(self.host, topic))
        tls_config = None
        if self.use_tls:
            tls_config = {
                "certfile": None,
                "keyfile": None,
                "cert_reqs": ssl.CERT_REQUIRED,
                "tls_version": ssl.PROTOCOL_TLSv1_2,
                "ciphers": None,
            }

        publish.single(
            topic,
            json.dumps(message),
            1,
            auth=self.auth,
            hostname=self.host,
            port=self.port,
            tls=tls_config,
        )


class HTTPClient:
    def __init__(self, url, username, password, method="POST"):
        self.url = url
        self.username = username
        self.password = password
        self.method = method
        if self.username is not None and self.password is not None:
            self.auth = (self.username, self.password)
        else:
            self.auth = None

    def send_message(self, message, filename=None, node_id=None, hostname=None):
        headers = {"Content-type": "application/json"}
        url = self.url
        if filename is not None:
            url = url.replace("${filename}", filename)
        if node_id is not None:
            url = url.replace("${node_id}", node_id)
        if hostname is not None:
            url = url.replace("${hostname}", hostname)
        log.info("Sending results to {}".format(url))

        if self.auth is not None:
            headers["Authorization"] = "Basic " + base64.b64encode(
                bytes(self.auth[0] + ":" + self.auth[1], "utf-8")
            ).decode("utf-8")
        try:
            response = requests.request(
                self.method, url, headers=headers, data=json.dumps(message)
            )
            if response.status_code == 200:
                log.info("Successfully sent results to {}".format(url))
                return True
            else:
                log.error(
                    "Failed to send results to {}, status code is {}".format(
                        url, response.status_code
                    )
                )
                return False
        except Exception as e:
            log.error(e)
            return False

