from bs4 import BeautifulSoup
from PIL import Image
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
import io
import os
import warnings
import base64
import pandas as pd


warnings.filterwarnings("ignore")


def pdf_to_json(file_path: str, min_words: int = 20) -> list:
    """将pdf文件转换为json list:
    file_path:pdf文件路径
    min_words:筛选掉词数小于min_words的元素
    返回json list
    """
    try:
        elements = partition_pdf(
            file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            split_pdf_page=True,
            split_pdf_allow_failed=True,
            split_pdf_concurrency_level=15,
            languages=["eng"],
        )
        output_list = [element.to_dict() for element in elements]
        output_list_modified = tables_from_html(output_list)
        output_list_modified = filter_words(output_list_modified, min_words=min_words)
        return output_list_modified
    except KeyError as e:
        # TODO 异常处理
        print(e)
        return e.message


def html_to_json(path: str, min_words: int = 20) -> list:
    """将html文件转换为json list:
    file_path:html文件路径
    min_words:筛选掉词数小于min_words的元素
    返回json list
    """
    try:
        if "http" in path:
            elements = partition_html(url=path)
        else:
            elements = partition_html(filename=path)
        output_list = [element.to_dict() for element in elements]
        return output_list
    except KeyError as e:
        # TODO 异常处理
        print(e)
        return e.message


def csv_to_json(file_path: str, min_words: int = 20) -> list:
    """将csv文件转换为json list:
    file_path:csv文件路径
    min_words:筛选掉词数小于min_words的元素
    返回json list
    """
    try:
        # 假设你的数据已经被读取到一个DataFrame中，命名为df
        df = pd.read_csv(file_path)  # 读取CSV文件
        output_list = df.to_dict(orient="records")  # 转换为字典
        return output_list
    except Exception as e:
        # TODO 异常处理
        print(e)
        return e


# TODO 图片处理
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


def filter_words(input_list: list, min_words: int) -> list:
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


class FileConverter:
    # TODO 写注释
    def __init__(self, paths):
        self.paths = paths

    def file_to_json(self):
        results = []
        for path in self.paths:
            print(f"Converting {path} to json...")
            _, ext = os.path.splitext(path)
            if ext.lower() == ".pdf":
                results.append(pdf_to_json(path))
            elif ext.lower() == ".html" or "http" in path:
                results.append(html_to_json(path))
            elif ext.lower() == ".csv":
                results.append(csv_to_json(path))
            else:
                print(f"Unsupported file type: {path}")
                results.append(None)
        return results

    def json_to_QA(self):
        pass


if __name__ == "__main__":
    file_converter = FileConverter(["Base.csv"])
    json = file_converter.file_to_json()
    print(json)
    with open("output.json", "w", encoding="utf-8") as f:
        f.write(str(json))
