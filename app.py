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
from yarl import URL
import logging
import time


logger = logging.getLogger(__name__)


_aws_lambda = True
if not _aws_lambda:
    from back import read_prescription

_MAX_SIZE = 2048

if "run_workflow" not in st.session_state:
    st.session_state.run_workflow = True

if "result_img" not in st.session_state:
    st.session_state.result_img = None

if "jwt" not in st.session_state:
    st.session_state.jwt = None


# Ikomia Scale URL
IKSCALE_URL=URL("https://scale.ikomia.net")

# Ikomia Scale project ID
PROJECT_ID="0185c004-b8a8-4758-a517-2e7e37f8defe"

# Ikomia Scale project deployment endpoint
ENDPOINT_URL = "https://583wuyhtl3.execute-api.eu-west-3.amazonaws.com"


class HTTPBadCodeError(Exception):
    """Raised when request status code <200 or >299."""

    def __init__(self, url: URL, code: int, content=None):
        """
        Init a new HTTP error.

        Args:
            url: URL
            code: HTTP code
            headers: Response header
            content: Response content
        """
        super().__init__(f"Bad return code {code} on '{url}'")
        self.url = url
        self.code = code
        self.content = content


def request(session, headers, method, url, data=None):

    if data is not None:
        data = json.dumps(data)

    request = requests.Request(
        method=method,
        url=url,
        headers=headers,
        #            params=query,
        data=data,
    )
    prepared_request = session.prepare_request(request)

    # Produce some debug logs
    logger.debug("Will %s '%s'", prepared_request.method, prepared_request.url)
    if prepared_request.headers:
        logger.debug(" with headers : %s", prepared_request.headers)
    if prepared_request.body:
        if len(prepared_request.body) > 10240:
            logger.debug(" with body    : .... too long to be dumped ! ...")
        else:
            logger.debug(" with body    : %s", prepared_request.body)

    try:
        response = session.send(
            prepared_request,
            allow_redirects=False,
            timeout=(30, 30),
        )
    except requests.exceptions.ReadTimeout as e:
        return None

    logger.debug("Response code : %d", response.status_code)
    logger.debug(" with headers : %s", response.headers)
    if ("Content-Length" in response.headers and int(response.headers["Content-Length"]) > 10240) or len(
        response.content
    ) > 10240:
        logger.debug(" with content    : .... too long to be dumped ! ...")
    else:
        logger.debug(" with content    : %s", response.content)

    if response.status_code >= 200 and response.status_code <= 299:
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.content

    raise HTTPBadCodeError(url, response.status_code, response.content)


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
        "inputs": [{"image": img_base64}],
        "outputs": [
            {"task_name": "read_eye_prescription", "task_index": 0, "output_index": 0},
            {"task_name": "read_eye_prescription", "task_index": 0, "output_index": 1},
            {"task_name": "read_eye_prescription", "task_index": 0, "output_index": 4}
        ],
        "parameters": [],
        "isBase64Encoded": True
    }

    #payload = json.dumps(body)
    #return headers, payload
    return body


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


def do(api_url, jwt, image):
    url = URL(api_url)

    session = requests.Session()
    headers = {
        "User-Agent": "IkomiaCli",
        "Authorization": f"Bearer {jwt}",
    }

    # Run workflow
    #response = request(session, headers, "PUT", url / "api/run", data=payload(task, task_parameters, image))
    response = request(session, headers, "PUT", url / "api/run", data=make_payload(image))

    # Get results
    uuid = response

    response = None

    while response is None or len(response) == 0:
        response = request(session, headers, "GET", url / f"api/results/{uuid}")
        time.sleep(5)

    return response


def process_image(url, jwt, image, img_display, values_display, title_values, transposed_values_display,
                  title_transposed_values, others_display):
    img_base64 = pil_image_to_b64str(image)

    if _aws_lambda:
        json_data = do(url, jwt, img_base64)
        for line in json_data:
            (t, d) = next(iter(line.items()))
            if t == "image":
                output_img = d
            elif t == "OUTPUT_GRAPHICS":
                graphics_output = d
            elif t == "DATA_DICT":
                data_dict = d
            else:
                raise ValueError(f"Can't parse {t} in response")
    else:
        data_dict, output_img, graphics_output = read_prescription(img_base64)
        data_dict = json.loads(data_dict)
        output_img = json.loads(output_img)["image"]
        graphics_output = json.loads(graphics_output)

    output_img = b64str_to_numpy(output_img)
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


def get_jwt(api_token, project_id):

    if st.session_state.jwt is None:
        session = requests.Session()
        headers = {
            "User-Agent": "IkomiaStreamlit",
            "Authorization": f"Token {api_token}",
        }

        response = request(session, headers, "GET", IKSCALE_URL / f"v1/projects/{project_id}/jwt/")

        if "id_token" in response:
            st.session_state.jwt = response["id_token"]
        else:
            raise Exception("Can't parse response {response}")

    return st.session_state.jwt


def demo():

    st.set_page_config(
        page_title="Ophthalmology Prescription Reader",
        page_icon="./images/ikomia_logo_400x400.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.ikomia.com",
            "About": f"This is 'Ophthalmology Prescription Reader' workflow",
        },
    )

    #
    #   Sidebar
    #
    st.sidebar.image("./images/ikomia_logo_400x400.png", width=100)

    api_token = st.sidebar.text_input("API Token")
    with st.sidebar.expander("API", expanded=False):
        project_id = st.text_input("Ikomia Project ID", value=PROJECT_ID)
        endpoint_url = st.text_input("API Endpoint URL", value=ENDPOINT_URL)

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
        # Get JWT from ikscale
        jwt = get_jwt(api_token, project_id)

        # Deployment Endpoint invocation
        with st.spinner("Wait for results..."):
            try:
                process_image(endpoint_url, jwt, input_img, img_display, values_display, title_values, transposed_values_display,
                              title_transposed_values, others_display)
            except HTTPBadCodeError as e:
                if e.code == 307:  # Temporary redirect to log in
                    st.session_state.jwt = None  # Purge stored JWT
                raise
    else:
        display_result(st.session_state.result_img, img_display, values_display, title_values,
                       transposed_values_display,
                       title_transposed_values, others_display)


if __name__ == '__main__':
    demo()
