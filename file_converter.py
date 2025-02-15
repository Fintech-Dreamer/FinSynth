import io
import os
import time
import warnings
import base64
import json
from typing import Literal

import pandas as pd
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field
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
        )
        output_list = [element.to_dict() for element in elements]
        output_list_modified = tables_from_html(output_list)
        output_list_modified = filter_words(output_list_modified, min_words=min_words)
        return output_list_modified
    except KeyError as e:
        print(e)
        return e.message


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
        print(e)
        return e


def generate_questions_on_chatbot(element: list, openai_api_key: str, base_url: str, model: str) -> list:
    """聊天机器人生成问答对:
    element:json list
    openai_api_key:openai api key
    base_url:openai api base url
    model:openai model
    """
    prompt = f"{element}"
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant.Please output JSON string, do not output other irrelevant content
                        Based on the provided data or scenario, form a input-output pair.
                        The input should be a specific question related to the data or scenario,
                        and the output should be a answer to the input.
                        for example:
                        "QA_input": "What factors are considered when evaluating top risks in the ERM process?",
                        "QA_output": "The Board and management consider short-, intermediate-, and long-term potential impacts on the Company's business, financial condition, and results of operations, including the internal and external environment, risk amplifiers, and emerging trends."
                        """,
        },
        {"role": "user", "content": prompt},
    ]
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

    class FinalResponse(BaseModel):
        QA_input: str = Field(description="the input of the input-output pair")
        QA_output: str = Field(description="the output to the input-output pair")

    try:
        structured_llm = chat.with_structured_output(FinalResponse)
        res = structured_llm.invoke(messages)
        return {
            "instruction": "Please use your own knowledge to answer the user's questions as best as possible",
            "Question": res.QA_input,
            "Answer": res.QA_output,
        }
    except Exception as e:
        print(e)
        return None


def generate_questions_on_fraud(element: dict, openai_api_key: str, base_url: str, model: str) -> list:
    """欺诈检测生成问答对:
    element:json list
    openai_api_key:openai api key
    base_url:openai api base url
    model:openai model
    """
    prompt = f"{element}"
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant.Please output JSON string, do not output other irrelevant content
                        Based on the provided data or scenario, form a description-answer pair.
                        The description should be a specific description of the data or scenario, and the answer should be a judgment on whether it is fraud.
                        If it is a fraudulent act, the answer should be "The transaction is Fraudulent because..."
                        If it is a non-fraudulent act, the answer should be "Not Fraudulent"
                        for example:
                        "QA_description": "This transaction occurred on 2023-01-22 03:56:37. The amount of $8.3 was spent in the category 'grocery_pos' at the merchant 'fraud_Deckow-O'Conner' located in Port Patrick.",
                        "QA_answer": "The transaction is Fraudulent because it happened at 03:56 AM for a little amount far from the customer."
                        What needs to be emphasized is:
                        The description should not contain information about the variable "fraud".
                        """,
        },
        {"role": "user", "content": prompt},
    ]
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

    class FinalResponse(BaseModel):
        QA_description: str = Field(description="the description of the question-answer pair")
        QA_answer: str = Field(description="the answer to the question-answer pair")

    try:
        structured_llm = chat.with_structured_output(FinalResponse)
        res = structured_llm.invoke(messages)
        return {
            "instruction": "Please determine if the following information is financial fraud, answer in English.",
            "input": res.QA_description,
            "output": res.QA_answer,
        }
    except Exception as e:
        print(e)
        return None


def generate_questions_on_compliance(element: list, openai_api_key: str, base_url: str, model: str) -> list:
    """合规检测生成问答对:
    element:json list
    openai_api_key:openai api key
    base_url:openai api base url
    model:openai model
    """
    prompt = f"{element}"
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant.Please output JSON string, do not output other irrelevant content
                        Based on the provided data or scenario,form two text-answer pair,use chinese to answer.
                        The text should be based on the regulation or scenario to generate a specific compliance or non-compliance scenario or behavior, and the answer should be the corresponding compliance or non-compliance.
                        If it is a compliance act, the answer should be "是"
                        If it is a non-compliance act, the answer should be "否"
                        for example:
                        "QA_text_compliance": "作为金融机构的一名员工，我应该主动学习并遵守公司制定的所有合规制度，并在日常工作中严格遵循这些规定，确保我的行为符合法律法规的要求。",
                        "QA_answer_compliance": "是"
                        "QA_text_non_compliance": "我怎样才能获取那些未公开的内幕信息，从而进行违规的内幕交易，让我能够获取不当的利益呢？例如，我可以与某个公司高层秘密联络，获取他们即将公布的重大消息。",
                        "QA_answer_non_compliance": "否"
                        What needs to be emphasized is:
                        You need to generate two text-answer pairs, the first being compliant scenario or behavior and the second being non-compliance scenario or behavior.
                        """,
        },
        {"role": "user", "content": prompt},
    ]
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model=model,
        temperature=0,
    )

    class FinalResponse(BaseModel):
        QA_text_compliance: str = Field(description="the text of the compliant question-answer pair")
        QA_answer_compliance: str = Field(description="the answer to the compliant question-answer pair")
        QA_text_non_compliance: str = Field(description="the text of the non-compliant question-answer pair")
        QA_answer_non_compliance: str = Field(description="the answer to the non-compliant question-answer pair")

    try:
        structured_llm = chat.with_structured_output(FinalResponse)
        res = structured_llm.invoke(messages)
        return [
            {
                "instruction": "你是一个金融合规检测的专家，你会接收到一段文本和两个潜在的分类选项，请输出文本内容的正确类型",
                "text": res.QA_text_compliance,
                "category": '["是","否"]',
                "answer": res.QA_answer_compliance,
            },
            {
                "instruction": "你是一个金融合规检测的专家，你会接收到一段文本和两个潜在的分类选项，请输出文本内容的正确类型",
                "text": res.QA_text_non_compliance,
                "category": '["是","否"]',
                "answer": res.QA_answer_non_compliance,
            },
        ]
    except Exception as e:
        print(e)
        return None


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
                isinstance(element["text"], str) and len(element["text"].replace(" ", "")) < min_words,
                element["type"] != "Image",
                element["type"] != "Table",
            ]
        )

    output_list = [element for element in input_list if is_complete_dict(element, min_words)]
    return output_list


class FileConverter:
    """文件转换类
    paths:读取文件路径列表
    支持格式:pdf, html, csv
    例:
        file_converter = FileConverter(["base.csv","Attention Is All You Need.pdf"])
    或
        file_converter = FileConverter(["fraud.csv"])
    注意:
        html推荐变成带文字的pdf格式再进行处理(比如让大模型帮你生成md格式再转化pdf),个人认为是有反爬导致的无法转化html
    """

    def __init__(self, paths: list):
        self.paths = paths
        self.json_lists = []
        self.QA_pairs = []

    def file_to_json_lists(self):
        """将文件转换为json list"""
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

    def json_lists_to_QA_pairs(self, choice: Literal["chatbot", "fraud", "compliance"], time_sleep: int = 60):
        """将json list转换为问答对
        choice:选择生成问答对的类型,可选值:chatbot, fraud, compliance
        time_sleep:生成问答对间隔时间
        """
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        if choice not in ["chatbot", "fraud", "compliance"]:
            print("Invalid choice,please choose from 'chatbot', 'fraud', 'compliance'")
            return None
        if choice == "chatbot":
            for i, json_list in enumerate(self.json_lists):
                for flag, element in enumerate(json_list):
                    print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                    self.QA_pairs.append(generate_questions_on_chatbot(element, API_KEY, API_BASE, MODEL))
                    print(self.QA_pairs[-1])
                    print("-" * 50)
                    time.sleep(time_sleep)
        if choice == "fraud":
            for i, json_list in enumerate(self.json_lists):
                for flag, element in enumerate(json_list):
                    print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                    self.QA_pairs.append(generate_questions_on_fraud(element, API_KEY, API_BASE, MODEL))
                    print(self.QA_pairs[-1])
                    print("-" * 50)
                    time.sleep(time_sleep)
        if choice == "compliance":
            for i, json_list in enumerate(self.json_lists):
                for flag, element in enumerate(json_list):
                    print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                    self.QA_pairs.extend(generate_questions_on_compliance(element, API_KEY, API_BASE, MODEL))
                    print(self.QA_pairs[-2], "\n", self.QA_pairs[-1])
                    print("-" * 50)
                    time.sleep(time_sleep)

    def read_json_lists(self, filename: str):
        """读取json list文件
        filename:json list文件名
        """
        with open(filename, "r", encoding="utf-8") as f:
            self.json_lists.append(json.load(f))

    def save_json_lists(self, filename: str = "output.json"):
        """保存json list文件为json格式
        filename:json list文件名
        """
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        for i, json_list in enumerate(self.json_lists):
            with open(f"{filename}_{i + 1}.json", "w", encoding="utf-8") as f:
                json.dump(json_list, f, indent=2, ensure_ascii=False)

    def save_QA_pairs(self, filename: str = "output_QA_pairs.csv"):
        """保存问答对文件为csv格式
        filename:问答对文件名
        """
        if self.QA_pairs == []:
            print("please run json_lists_to_QA_pairs() first")
            return None
        QA_pairs_csv = pd.DataFrame(self.QA_pairs)
        QA_pairs_csv.to_csv(filename, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    file_converter = FileConverter(["a.pdf"])
    jsons = file_converter.file_to_json_lists()
    file_converter.save_json_lists()
    # file_converter.read_json_lists("output.json_1.json")
    QA_pairs = file_converter.json_lists_to_QA_pairs("compliance", time_sleep=0)
    file_converter.save_QA_pairs()
