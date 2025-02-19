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

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from typing import Tuple,List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
import chromadb  # 导入Chroma客户端库
from langchain_text_splitters import CharacterTextSplitter
from params import API_KEY, API_BASE, MODEL,MODEL_PICTURE
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


def generate_questions_on_chatbot(element: list, openai_api_key: str, base_url: str, model: str,picture_model:str) -> list:
    """聊天机器人生成问答对:
    element:json list
    openai_api_key:openai api key
    base_url:openai api base url
    model:openai model
    """
    # 判断是否有图片，有图片会生成摘要来代替图片
    img_base64 = element["metadata"]["image_base64"] if "image_base64" in element["metadata"] else ""
    img_texts = image_summarize(img_base64,openai_api_key, base_url,picture_model) if img_base64 else ""
    #将文本和图片摘要合并
    background = str([element["text"],img_texts])
    prompt = f"{background}"
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
            "background": element["text"],
            "img_texts": img_texts,
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

    def json_lists_to_QA_pairs(self, choice: Literal["chatbot", "fraud", "compliance"], time_sleep: int = 60,lable: int = -1):
        """将json list转换为问答对
        choice:选择生成问答对的类型,可选值:chatbot, fraud, compliance
        time_sleep:生成问答对间隔时间
        lable:选择json list的前几个元素,默认为-1,即全部元素
        """
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        if lable != -1:
            self.json_lists = [self.json_lists[0][:lable]]
        if choice not in ["chatbot", "fraud", "compliance"]:
            print("Invalid choice,please choose from 'chatbot', 'fraud', 'compliance'")
            return None
        if choice == "chatbot":
            for i, json_list in enumerate(self.json_lists):
                for flag, element in enumerate(json_list):
                    if len(element["text"]) <= 200 or len(element["text"]) >= 5000:#过滤掉长度不符合要求的元素
                        continue
                    print(f"Converting {self.paths[i]} element {flag + 1} to QA pairs...")
                    self.QA_pairs.append(generate_questions_on_chatbot(element, API_KEY, API_BASE, MODEL,MODEL_PICTURE))
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
    def json_lists_to_QA_pairs_rag(self, choice: Literal["chatbot"], time_sleep: int = 60,lable: int = -1,embed_model=None):
        """将json list转换为问答对
        choice:选择生成问答对的类型,可选值:chatbot
        time_sleep:生成问答对间隔时间
        lable:选择json list的前几个元素,默认为-1,即全部元素
        """
        if self.json_lists == []:
            print("please run file_to_json_lists() first")
            return None
        if lable != -1:
            self.json_lists = [self.json_lists[0][:lable]]
        if choice not in ["chatbot"]:
            print("Invalid choice,please choose from 'chatbot'")
            return None
        if choice == "chatbot":#批量一次性处理10个元素块
            while len(self.json_lists[0]) >= 10:
                # 合并10个元素块的文本和图片摘要，顺序融合，并且过滤较短的文本块
                json_lists_use = [block for block in self.json_lists[0] if len(block["text"]) > 200]
                texts_4k_token = [x["text"] for x in json_lists_use]
                images_list_base64=[x["metadata"]['image_base64'] if 'image_base64' in x["metadata"] else "" for x in json_lists_use]
                images_list=[image_summarize(image_base64,API_KEY, API_BASE,MODEL_PICTURE) if image_base64 else "" for image_base64 in images_list_base64]
                context_list=[str(texts_4k_token[i]+images_list[i]) for i in range(len(json_lists_use))] 
                #context转成向量库
                vectorstore=vectorstore_create(context_list,embed_model)
                #生成问题，每个元素块生成一个问题
                self.json_lists[0] = self.json_lists[0][10:] if len(self.json_lists[0]) > 10 else []
                questions_list=get_questions_only(texts_4k_token)  
                #文本向量库检索s
                Q_B_list=[]
                Q_B_list=[{"question":query,"docs_list":"\n".join(text_retriever(query,vectorstore))} for query in questions_list]
                #生成答案
                answer_list=generate_answer_only(Q_B_list)
                #生成问题答案对
                QA_list=[]
                QA_list=[{"Question":Q_B_list[i]["question"],"Answer":answer_list[i],"Background":Q_B_list[i]["docs_list"]} for i in range(len(Q_B_list))]
                self.QA_pairs.extend(QA_list)
                #删除向量库,避免出现重复
                vectorstore=delete_vectorstore(vectorstore)
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

def image_summarize(img_base64, openai_api_key: str, base_url: str, model_picture: str)->str:#jpg格式
    """Make image summary"""
    chat = ChatOpenAI(
    api_key=openai_api_key,
    base_url=base_url,
    model=model_picture,
    temperature=0
    )
    #构造提示词
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def get_questions_only(texts_4k_token:list)->list:

    class Entities(BaseModel):
        """generated questions"""

        questions: List[str] = Field(
            ...,
            description="""The generated questions.""",
        )

    summarys=str(texts_4k_token)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Generate a question for each element in the input list, 
                preferably related to financial analysis,  Format your response as a list of 'Question: ... ' pairs,
                and output in JSON format.""",
            ),
            (
                "human",
                """"input: {summarys}""",
            ),
        ]
    )

    llm = ChatOpenAI(
            openai_api_key=API_KEY,
            base_url=API_BASE,
            model=MODEL,
            temperature=0
        )

    entity_chain = prompt | llm.with_structured_output(Entities)
    


    question_lists=entity_chain.invoke({"summarys": summarys})
    return question_lists.questions

def vectorstore_create(context:list,embed_model)->Chroma:
    text_splitter = CharacterTextSplitter(
    separator=".",#因为分割对象是英文文本
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
    )
    texts = ".".join(context)
    context_list = text_splitter.split_text(texts)
    #加载vectorstore
    vectorstore = Chroma.from_texts(texts=context_list, embedding=embed_model , collection_name="BAAI")
    return vectorstore
def delete_vectorstore(vectorstore:Chroma):
    # 获取Chroma客户端并删除已存在的集合
    client = chromadb.Client()
    client.delete_collection("BAAI")  # 确保集合名称正确
    return vectorstore
#检索支撑事实（后续可以加上上下两个块，来扩大检索范围）
def text_retriever(query:str,vectorstore)->list[str]:
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.25})
    # docs = retriever.get_relevant_documents(query)
    # 利用个数来筛选
    docs = vectorstore.similarity_search(query,k = 2)
    if docs==[]:
        docs_list=[]
    else:
        docs_list=list(x.page_content for x in docs)
    return docs_list

def generate_answer_only(Q_B_list:list[str,str])->list[str,str]:

    prompt_list=[f"""Give me the simple answer based on the question and  background 
    Question:{Q_B["question"]}
    Background:{Q_B["docs_list"]}
    Notice:Your output format must be answer:
    """ for Q_B in Q_B_list]

    chat = ChatOpenAI(
        openai_api_key=API_KEY,
        base_url=API_BASE,
        model=MODEL,
        temperature=0
    )
    messages_list=[[
        {"role": "system", "content": "You are a helpful assistant. Generate the Answer based on the Question and  Background"},
        {"role": "user", "content": prompt}
    ]  for prompt in prompt_list]
    
    # 调用 batch 方法并设置并发数
    responses = chat.batch(messages_list, config={"max_concurrency": 5})
    return [response.content for response in responses]

if __name__ == "__main__":
    file_converter = FileConverter(["NVIDIA54.pdf"])
    # jsons = file_converter.file_to_json_lists()
    # file_converter.save_json_lists()
    file_converter.read_json_lists("NVIDIA.json")
    #加载embeddings model
    embed_model = HuggingFaceBgeEmbeddings(model_name=r"E:\model\BAAI\bge-m3")#model_name存放embeddings的路径
    QA_pairs = file_converter.json_lists_to_QA_pairs_rag("chatbot", time_sleep=0, lable=10,embed_model=embed_model)

    #QA_pairs = file_converter.json_lists_to_QA_pairs("chatbot", time_sleep=0, lable=5)
    file_converter.save_QA_pairs()
