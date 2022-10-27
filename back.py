from ikomia.core import task
import numpy as np
import base64
from PIL import Image
import io


def b64str_to_numpy(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytes_obj = io.BytesIO(base64bytes)
    img = Image.open(bytes_obj)
    return np.asarray(img)


def read_prescription(img):
    t = task.create("read_eye_prescription")
    input_img = t.getInput(0)
    input_img.setImage(b64str_to_numpy(img))
    t.run()
    ik_output = t.getOutput(4)
    data_dict = ik_output.toJson()
    output_img = t.getOutput(0).toJson(['image_format', 'jpg'])
    graphics_output = t.getOutput(1).toJson([])
    return data_dict, output_img, graphics_output


