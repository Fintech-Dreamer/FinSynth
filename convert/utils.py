import os
from PIL import Image
import io
import base64
from bs4 import BeautifulSoup


def base64_to_image(output_list: list, output_path: str):
    """将base64字符串转换为图片并保存到指定路径:
    resp:从unstructured_client库获取的json list
    output_path:输出路径
    """
    for element in output_list:
        if "image_base64" in element["metadata"]:
            image_data = base64.b64decode(element["metadata"]["image_base64"])
            image = Image.open(io.BytesIO(image_data))
            image.save(os.path.join(output_path, f"image_{element['element_id']}.png"))


def tables_from_html(output_list: list) -> list:
    """将属性为Table的元素的html数据转换为表格数据并添加到元素的text属性中:
    output_list:从unstructured_client库获取的json list
    返回json list
    """
    for element in output_list:
        if "Table" in element["type"]:
            html_data = element["metadata"]["text_as_html"]
            soup = BeautifulSoup(html_data, "html.parser")
            table_data = []
            rows = soup.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                cols = [col.text.strip() for col in cols]
                if cols:
                    table_data.append(cols)
            element["text"] = table_data
    return output_list


def filter_words(input_list: list, min_words: int = 20) -> list:
    """用来删选dict
    input_list:从unstructured_client库获取的json list
    min_words:最少词数
    is_complete_dict():判断是否为复合要求的dict
    返回json list
    """

    def is_complete_dict(element: dict, min_words) -> bool:
        return not all(
            [
                isinstance(element["text"], str) and len(element["text"].split()) < min_words,
                element["type"] != "Image",
                element["type"] != "Table",
            ]
        )

    output_list = [element for element in input_list if is_complete_dict(element, min_words)]
    return output_list
