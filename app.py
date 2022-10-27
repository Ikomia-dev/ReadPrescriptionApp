import streamlit as st
import numpy as np
import base64
import requests
import json
import io
from PIL import Image
from io import BytesIO
import pandas as pd
import cv2

from back import read_prescription

_MAX_SIZE = 2048

if "run_workflow" not in st.session_state:
    st.session_state.run_workflow = True

if "result_img" not in st.session_state:
    st.session_state.result_img = None


def on_input_change():
    st.session_state.run_workflow = True


def on_download_clicked():
    st.session_state.run_workflow = False


def pil_image_to_b64str(im):
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = img_bytes.decode('utf-8')
    return img_str


def b64str_to_numpy(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytes_obj = io.BytesIO(base64bytes)
    img = Image.open(bytes_obj)
    return np.asarray(img)


def bytesio_obj_to_b64str(bytes_obj):
    return base64.b64encode(bytes_obj.read()).decode("utf-8")


def bytesio_to_pil_image(bytes_obj):
    data = io.BytesIO(bytes_obj.read())
    img = Image.open(data)
    return img


def numpy_to_pil_image(np_img):
    if np_img.dtype in ["float32", "float64"]:
        return Image.fromarray((np_img * 255 / np.max(np_img)).astype('uint8'))
    else:
        return Image.fromarray(np_img)


def check_image_size(img):
    max_size = max(img.size)
    if max_size > _MAX_SIZE:
        if img.width > img.height:
            img = img.resize((_MAX_SIZE, int(img.height * _MAX_SIZE / img.width)))
        else:
            img = img.resize((int(img.width * _MAX_SIZE / img.height), _MAX_SIZE))

    return img


def make_payload(img_base64):
    headers = {"Content-Type": "application/json"}

    body = {
        "inputs": [
            {"type": "IMAGE", "data": img_base64},
        ],
        "outputs": [
            {"task_name": "read_eye_prescription", "task_index": 0, "output_index": 0},
            {"task_name": "read_eye_prescription", "task_index": 0, "output_index": 1},
            {"task_name": "read_eye_prescription", "task_index": 0, "output_index": 4}
        ],
        "parameters": [],
        "isBase64Encoded": True
    }

    payload = json.dumps(body)
    return headers, payload


def colorize_rows(s):
    return ['background-color: #c5b4e3'] * len(s) if s.name == 'Left' else ['background-color: #add8e6'] * len(s)


def display_result(results, output_img, graphics_output, img_display, values_display, title_values,
                   transposed_values_display, title_transposed_values, others_display):
    st.session_state.output_img = output_img
    img_display.image(np.asarray(output_img), use_column_width="auto", clamp=True)
    st.session_state.results = results
    values = list(results["values"].values())
    values = [["{:.2f}".format(v) if isinstance(v, float) else v for v in e] for e in values]
    transposed_values = list(results["transposed_values"].values())
    transposed_values = [["{:.2f}".format(v) if isinstance(v, float) else v for v in e] for e in transposed_values]
    title_values.header("Eye values")

    df_values = pd.DataFrame(values, index=pd.Index(["Right", "Left"]),
                             columns=["Sphere", "Cylinder", "Axe", "Add"])

    df_values = df_values.style.apply(colorize_rows, axis=1).set_properties(**{'font-size': '30pt'})

    values_display.table(df_values)
    title_transposed_values.header("Transposed eye values")
    df_transposed_values = pd.DataFrame(transposed_values, index=pd.Index(["Right", "Left"]),
                                        columns=["Sphere", "Cylinder", "Axe", "Add"])
    df_transposed_values = \
        df_transposed_values.style.apply(colorize_rows, axis=1).set_properties(**{'font-size': '30pt'})
    transposed_values_display.table(df_transposed_values)

    str_to_write = "## RPPS : {}\n## ADELI : {}\n## DR : {} {}".format(results["rpps"], results["adeli"],
                                                                       results["dr_name"].upper(),
                                                                       results["dr_first_name"].upper())
    others_display.markdown(str_to_write)


async def put(session, url, img_base64, bck_type, **kwargs):
    headers, payload = make_payload(img_base64, bck_type)
    response = await session.request('PUT', url=url, headers=headers, data=payload, **kwargs)
    data = await response.json(content_type=None)
    return data


def process_image(url, image, img_display, values_display, title_values, transposed_values_display,
                  title_transposed_values, others_display):
    img_base64 = pil_image_to_b64str(image)
    # headers, payload = make_payload(img_base64)
    # response = requests.put(url, headers=headers, data=payload)
    # data_dict = response.json()
    data_dict, output_img, graphics_output = read_prescription(img_base64)
    data_dict = json.loads(data_dict)
    output_img = json.loads(output_img)
    graphics_output = json.loads(graphics_output)
    output_img = b64str_to_numpy(output_img["image"])
    output_img = np.array(output_img[:, :, ::-1], dtype='uint8')
    for rect in graphics_output["items"]:
        x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
        pen_prop = rect["properties"]["pen"]
        color = [pen_prop["r"], pen_prop["g"], pen_prop["b"]]
        pt1 = [int(x), int(y)]
        pt2 = [int(x + w), int(y + h)]
        cv2.rectangle(output_img, pt1, pt2, color, 3)
    # Update image in streamlit view
    display_result(data_dict, output_img, graphics_output, img_display, values_display, title_values,
                   transposed_values_display,
                   title_transposed_values, others_display)


def demo():
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.image("./images/ikomia_logo_400x400.png", use_column_width=True)

    # Title
    st.title("Ophthalmology Prescription Reader")

    col1, col2 = st.columns([1, 1])

    # Column 1: source image
    uploaded_input = col1.file_uploader("Choose input image", on_change=on_input_change)
    img_display = col1.empty()
    title_values = col2.empty()
    values_display = col2.empty()
    title_transposed_values = col2.empty()
    transposed_values_display = col2.empty()
    others_display = col2.empty()

    # Display input image
    if uploaded_input is not None:
        input_img = bytesio_to_pil_image(uploaded_input)
        input_img = check_image_size(input_img)
        img_display.image(np.asarray(input_img), use_column_width="auto", clamp=True)
    else:
        return

    if st.session_state.run_workflow:
        # Local invocation
        # url = "http://localhost:9000/2015-03-31/functions/function/invocations"
        # AWS Lambda invocation
        url = "https://psjzth8q2l.execute-api.eu-west-1.amazonaws.com/"
        process_image(url, input_img, img_display, values_display, title_values, transposed_values_display,
                      title_transposed_values, others_display)
    else:
        display_result(st.session_state.result_img, img_display, values_display, title_values,
                       transposed_values_display,
                       title_transposed_values, others_display)


if __name__ == '__main__':
    demo()
