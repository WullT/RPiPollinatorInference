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
            "score": self.score,
            "width": self.width,
            "height": self.height,
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
            "score": self.score,
            "width": self.width,
            "height": self.height,
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
        self.process_time = None
        self.download_time = None
        self.original_image_size = None
        self.conf_threshold = None
        self.iou_threshold = None
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
            self.timestamp = datetime.datetime.fromisoformat(meta["capture_time"])
            self.process_time = meta["time_process"]
            self.download_time = meta["time_download"]
            self.original_image_size = meta["capture_size"]
            self.conf_threshold = meta["conf_thres"]
            self.iou_threshold = meta["iou_thres"]
            self.margin = meta.get("margin", None)
            self.model_name = meta.get("model_name", None)
            for i in range(len(msg["detections"])):
                self.classes.append(msg["detections"][i]["class"])
                self.scores.append(msg["detections"][i]["score"])
                self.images.append(self._load_image(msg["detections"][i]["image"]))
            self.num_detections = len(msg["detections"])
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
        self.process_time = None
        self.download_time = None
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
        metadata["inference_time"] = self.process_time
        metadata["capture_time"] = self.download_time
        metadata["confidence_threshold"] = self.conf_threshold
        metadata["iou_threshold"] = self.iou_threshold
        metadata["model_name"] = self.model_name
        metadata["original_image_size"] = self.original_image_size
        metadata["margin"] = self.margin
        return metadata

    def print_detections(self):
        print("Node ID: {}".format(self.node_id))
        print("Timestamp: {}".format(self.timestamp))
        print("Process time: {}".format(self.process_time))
        print("Download time: {}".format(self.download_time))
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
            "flowers": flowers,
            "pollinators": pollinators,
            "timestamp": str(self.timestamp),
            "node_id": self.node_id,
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
        with open(filepath + self.generate_filename(), "w") as f:
            json.dump(self.generate_message(save_crop=save_crop), f)
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
            topic.replace("${filename}", filename)
        if node_id is not None:
            topic.replace("${node_id}", node_id)
        if hostname is not None:
            topic.replace("${hostname}", hostname)
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
    def __init__(self, url,username, password, method="POST"):
        self.url = url
        self.username = username
        self.password = password
        self.method = method
        if self.username is not None and self.password is not None:
            self.auth = (self.username, self.password)
        else:
            self.auth = None

    def send_message(self, message, filename=None, node_id=None, hostname=None):
        headers = {'Content-type': 'application/json'}
        url = self.url
        if filename is not None:
            url.replace("${filename}", filename)
        if node_id is not None:
            url.replace("${node_id}", node_id)
        if hostname is not None:
            url.replace("${hostname}", hostname)
        log.info("Sending message to {}".format(self.url))

        if self.auth is not None:
            headers['Authorization'] = 'Basic ' + base64.b64encode(bytes(self.auth[0] + ':' + self.auth[1], 'utf-8')).decode('utf-8')
        try:
            response = requests.request(self.method, url, headers=headers, data=json.dumps(message))
            return response.status_code == 200
        except Exception as e:
            print(e)
            return False

