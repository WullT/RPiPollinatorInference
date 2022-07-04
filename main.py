import logging
import sys

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
log.addHandler(handler)

logging.basicConfig(level=logging.INFO)


import zmq
import time
import os
from PIL import Image
import base64
import yaml
import argparse
from yolomodelhelper import YoloModel
from messagehelper import (
    MessageParser,
    Flower,
    Pollinator,
    MessageGenerator,
    MQTTClient,
    HTTPClient,
)
import socket

parser = argparse.ArgumentParser(description="ZMQ Message Queue")
parser.add_argument("--config", type=str, default="config.yaml", help="config file")
args = parser.parse_args()
# parse yaml configuration file
with open(args.config, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        log.error(exc)
        exit(1)

HOSTNAME = socket.gethostname()
if "ap-" in HOSTNAME:
    HOSTNAME = HOSTNAME.replace("ap-", "")

model_config = config.get("model")
WEIGHTS_PATH = model_config.get("weights_path")
LOCAL_YOLOV5_PATH = model_config.get("local_yolov5_path")
CONFIDENT_THRESHOLD = model_config.get("confidence_threshold", 0.25)
IOU_THRESHOLD = model_config.get("iou_threshold", 0.45)
MARGIN = model_config.get("margin", 40)
MULTI_LABEL = model_config.get("multi_label", False)
MULTI_LABEL_IOU_THRESHOLD = model_config.get("multi_label_iou_threshold", 0.5)
MAX_DETECTIONS = model_config.get("max_detections", 10)
CLASS_NAMES = model_config.get("class_names")
AUGMENT = model_config.get("augment", False)


zmq_config = config.get("zmq")
ZMQ_HOST = zmq_config.get("host")
ZMQ_PORT = zmq_config.get("port")
ZMQ_REQ_TIMEOUT = zmq_config.get("request_timeout", 3000)
ZMQ_REQ_RETRIES = zmq_config.get("request_retries", 10)

STORE_FILE = False
BASE_DIR = "output"
SAVE_CROPS = True

output_config = config.get("output")
if output_config.get("file") is not None:
    output_config_file = output_config.get("file")
    if output_config_file.get("store_file", False):
        STORE_FILE = True
        BASE_DIR = output_config_file.get("base_dir", "output")
        SAVE_CROPS = output_config_file.get("save_crops", True)
        log.info("store_file is enabled, base_dir: {}".format(BASE_DIR))

TRANSMIT_MQTT = False
mclient = None
if output_config.get("mqtt") is not None:
    output_config_mqtt = output_config.get("mqtt")
    if output_config_mqtt.get("transmit_mqtt", False):
        TRANSMIT_MQTT = True
        log.info("Transmitting to MQTT")
        mqtt_host = output_config_mqtt.get("host")
        mqtt_port = output_config_mqtt.get("port")
        mqtt_topic = output_config_mqtt.get("topic")
        mqtt_topic = mqtt_topic.replace("${hostname}", HOSTNAME)
        mqtt_username = output_config_mqtt.get("username")
        mqtt_password = output_config_mqtt.get("password")
        mqtt_use_tls = output_config_mqtt.get("use_tls", mqtt_port == 8883)
        log.info(
            "MQTT host: {}, port: {}, topic: {}, username {} use_tls: {}".format(
                mqtt_host, mqtt_port, mqtt_topic, mqtt_username, mqtt_use_tls
            )
        )
        mclient = MQTTClient(
            mqtt_host, mqtt_port, mqtt_topic, mqtt_username, mqtt_password, mqtt_use_tls
        )

TRANSMIT_HTTP = False
hclient = None
if output_config.get("http") is not None:
    output_config_http = output_config.get("http")
    if output_config_http.get("transmit_http", False):
        TRANSMIT_HTTP = True
        log.info("Transmitting to HTTP")
        http_url = output_config_http.get("url")
        http_url = http_url.replace("${hostname}", HOSTNAME)
        http_username = output_config_http.get("username")
        http_password = output_config_http.get("password")
        http_method = output_config_http.get("method", "POST")
        log.info(
            "HTTP url: {}, method: {}, username: {}".format(
                http_url, http_method, http_username
            )
        )
        hclient = HTTPClient(http_url, http_username, http_password, http_method)

context = zmq.Context().instance()
log.info("Connecting to ZMQ server on tcp://{}:{}".format(ZMQ_HOST, ZMQ_PORT))
client = context.socket(zmq.REQ)
client.connect("tcp://{}:{}".format(ZMQ_HOST, ZMQ_PORT))


def request_message(code, client):
    """
    request codes:
        0: get first message
        1: get first message and remove it from queue
        2: remove first message from queue
    response:
        dict with message or
        response codes:
            0: no data available
            1: first message removed from queue
    """
    log.info("Sending request with code 1".format(code))
    client.send_json(code)
    retries_left = ZMQ_REQ_RETRIES
    while True:
        if (client.poll(ZMQ_REQ_TIMEOUT) & zmq.POLLIN) != 0:
            reply = client.recv_json()

            # print("Server replied (%s)", type(reply))
            return reply
        retries_left -= 1
        log.warning("No response from server")
        client.setsockopt(zmq.LINGER, 0)
        client.close()

        if retries_left == 0:
            log.error("Server seems to be offline, abandoning")
            exit(1)
        log.info("Reconnecting to server…")
        # Create new connection
        client = context.socket(zmq.REQ)
        client.connect("tcp://{}:{}".format(ZMQ_HOST, ZMQ_PORT))

        log.info("Resending code {}".format(code))
        client.send_json(code)


parser = MessageParser()


model = YoloModel(
    WEIGHTS_PATH,
    LOCAL_YOLOV5_PATH,
    confidence_threshold=CONFIDENT_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
    margin=MARGIN,
    multi_label=MULTI_LABEL,
    multi_label_iou_threshold=MULTI_LABEL_IOU_THRESHOLD,
    class_names=CLASS_NAMES,
    augment=AUGMENT,
    max_det=MAX_DETECTIONS,
)


while True:
    msg = request_message(1, client)  # get first message, remove it from queue
    if type(msg) == dict:
        message_ok = parser.parse_message(msg)
        if message_ok:
            generator = MessageGenerator()
            generator.set_timestamp(parser.timestamp)
            generator.set_node_id(parser.node_id)
            # parser.print_detections()
            log.info(
                "Got data from {}, recorded at {}, contains {} flowers".format(
                    parser.node_id, parser.timestamp, parser.num_detections
                )
            )

            if parser.num_detections > 0:
                model.reset_inference_times()
                pollinator_index = 0

                for flower_index in range(len(parser.images)):
                    image = parser.images[flower_index]
                    width, height = image.size
                    flower_obj = Flower(
                        index=flower_index,
                        class_name=parser.classes[flower_index],
                        score=parser.scores[flower_index],
                        width=width,
                        height=height,
                    )
                    generator.add_flower(flower_obj)
                    res = model.predict(image)
                    crops = model.get_crops()
                    boxes = model.get_boxes()
                    classes = model.get_classes()
                    scores = model.get_scores()
                    names = model.get_names()

                    pollinator_indexes = model.get_indexes()
                    # log.info("pollinator_indexes: {}".format(pollinator_indexes))
                    for detection in range(len(crops)):
                        idx = pollinator_index + pollinator_indexes[detection]
                        crop_image = Image.fromarray(crops[detection])
                        width, height = crop_image.size

                        pollinator_obj = Pollinator(
                            index=idx,
                            flower_index=flower_index,
                            class_name=names[detection],
                            score=scores[detection],
                            width=width,
                            height=height,
                            crop=crop_image,
                        )
                        generator.add_pollinator(pollinator_obj)
                    if len(pollinator_indexes) > 0:
                        pollinator_index += max(pollinator_indexes) + 1
            log.info(
                "Inference times [total, avg]: {}".format(model.get_inference_times())
            )
            model_meta = model.get_metadata()
            generator.add_metadata(model_meta, "pollinator_inference")
            generator.add_metadata(parser.get_metadata(), "flower_inference")
            # log.info("Model metadata")
            # log.info(json.dumps(model_meta, indent=4))

            res_msg = generator.generate_message()
            if STORE_FILE:
                generator.store_message(BASE_DIR, SAVE_CROPS)
            if TRANSMIT_MQTT:
                mclient.publish(res_msg)
            if TRANSMIT_HTTP:
                hclient.send_message(res_msg)

    elif type(msg) == int:
        if msg == 0:  # no data available
            log.info("No data available")
            time.sleep(5)
