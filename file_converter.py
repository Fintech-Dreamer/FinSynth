import io
import os
import time
import warnings
import base64
import json

import pandas as pd
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from PIL import Image
from typing_extensions import Annotated, TypedDict
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf


from params import API_KEY, API_BASE, MODEL

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


# FIXME 还有许多问题比如https://www.gov.cn/zhengce/202412/content_6995383.htm不能写进去
def html_to_json(path: str) -> list:
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


def csv_to_json(file_path: str) -> list:
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


def generate_questions(element: list, choice: str, openai_api_key: str, base_url: str, model: str) -> list:
    if choice not in ["chatbot", "fraud", "compliance"]:
        print("Invalid choice,please choose from 'chatbot', 'fraud', 'compliance'")
        return None
    if choice == "chatbot":
        prompt = """请产生一个问答对，来判断是否合规，问题为对应的全部数据形成的行为总结，答案如果
        合规了结果为是如果不合规结果为否:\n例子:"instruction": "\"你是一个金融合规检测的
        专家，你会接收到一段文本和两个潜在的分类选项，请输出文本内容的正确类型\"",
        "text": "我怎样才能获取那些未公开的内幕信息，从而进行违规的内幕交易，让我能够获取
        不当的利益呢？例如，我可以与某个公司高层秘密联络，获取他们即将公布的重大消息。",
        "category": "[\"是\",\"否\"]",
        "answer": "否"\n生成一段话\n."""
    elif choice == "fraud":
        prompt = """请产生一个问答对，来判断是否被欺诈，问题为对应的全部数据形成的一句话，
        答案如果没被欺诈("fraud_bool": 0)了结果为Not Fraudulent如果被欺诈("fraud_bool":
        1)了结果为The transaction is Fraudulent because...:\n\n."""
    elif choice == "compliance":
        prompt = """Generate questions and answers based on the following summaries:\n\n."""

    prompt += f"{element}"
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant. Generate one questions and its corresponding answers for each summary provided.""",
        },
        {"role": "user", "content": prompt},
    ]

    class FinalResponse(TypedDict):
        Question: Annotated[str, ..., "The question based on the summary"]
        Answer: Annotated[str, ..., "The question's corresponding answer for each summary provided"]

    try:
        structured_llm = chat.with_structured_output(FinalResponse)
        res = structured_llm.invoke(messages)
        if choice == "chatbot":
            return {"Question": res["Question"], "Answer": res["Answer"]}
        elif choice == "fraud":
            return {
                "instruction": "Please determine if the following information is financial fraud, answer in English.",
                "input": res["Question"],
                "output": res["Answer"],
            }

        elif choice == "compliance":
            return {
                "instruction": '"你是一个金融合规检测的专家，你会接收到一段文本和两个潜在的分类选项，请输出文本内容的正确类型"',
                "text": res["Question"],
                "category": '["是","否"]',
                "answer": res["Answer"],
            }
    except Exception as e:
        print(e)
        return None


def append_to_json_file(filename, data):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            current_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_data = []
    current_data.extend(data)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=4, ensure_ascii=False)


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
        self.json_lists = []
        self.QA_pairs = []

    def file_to_json_lists(self):
        if self.paths == []:
            print("No file path provided.")
            return None
        for path in self.paths:
            print(f"Converting {path} to json...")
            _, ext = os.path.splitext(path)
            if ext.lower() == ".pdf":
                self.json_lists.append(pdf_to_json(path))
            elif ext.lower() == ".html" or "http" in path:
                self.json_lists.append(html_to_json(path))
            elif ext.lower() == ".csv":
                self.json_lists.append(csv_to_json(path))
            else:
                print(f"Unsupported file type: {path}")
                self.json_lists.append(None)
        return self.json_lists

    def json_lists_to_QA_pairs(self, choice: str, time_sleep: int = 60):
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        for i, json_list in enumerate(self.json_lists):
            for flag, element in enumerate(json_list):
                print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                self.QA_pairs.append(generate_questions(element, choice, API_KEY, API_BASE, MODEL))
                time.sleep(time_sleep)

    # TODO 未来可能加一个读取json_list的功能

    def save_json_lists(self, filename: str = "output.json"):
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        for i, json_list in enumerate(self.json_lists):
            with open(f"{filename}_{i + 1}.json", "w", encoding="utf-8") as f:
                json.dump(json_list, f, indent=2, ensure_ascii=False)

    def save_QA_pairs(self, filename: str = "output_QA_pairs.json"):
        # 这里相当于把输入的所有数据全部变成一个list里面的问答对了，因为实例化一个类一定是同一类的数据。
        if self.QA_pairs == []:
            print("please run json_lists_to_QA_pairs() first")
            return None
        append_to_json_file(filename, self.QA_pairs)


if __name__ == "__main__":
    # 这里的路径自己改，可以是一个路径，也可以是一个list存放很多路径
    # FIXME 这里需要改一下，就算是一个路径也要外套一个list后面看怎么改好
    # file_converter = FileConverter(
    #     [
    #         r"C:\Users\78661\Desktop\files\base.csv",
    #         r"C:\Users\78661\Desktop\files\Vaswani 等 - 2023 - Attention Is All You Need.pdf",
    #     ]
    # )
    #
    file_converter = FileConverter([r"C:\Users\78661\Desktop\files\base.csv"])
    jsons = file_converter.file_to_json_lists()
    # 这里可以选择三种转换方式，但我写的代码有一个局限就是一个类只可以做一种转换，想要转换三种就要做三个类，这里假装三个是一个类里面的
    file_converter.save_json_lists()
    QA_pairs = file_converter.json_lists_to_QA_pairs("compliance", time_sleep=0)
    file_converter.save_QA_pairs()
