import os
import base64

def encode_image_to_base64(image_path):
    """
    Read an image file and encode it as a base64 string.

    :param image_path: Local path to the image file
    :return: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')