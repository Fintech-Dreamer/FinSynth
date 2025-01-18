from utils import tables_from_html, filter_words
import warnings
import os
import json
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

warnings.filterwarnings("ignore")

# 登录
client = UnstructuredClient(
    api_key_auth="UlcNeEb8IbwAxS4uZrbPJj30rJonRb",
    server_url="https://api.unstructuredapp.io",
)

# 遍历convert下的input下的pdf文件夹下的pdf
for file in os.listdir("convert/input/pdf"):
    if file.endswith(".pdf"):
        file_path = f"convert/input/pdf/{file}"
        print(f"Processing {file}...")
        req = {
            "partition_parameters": {
                "files": {
                    "content": open(file_path, "rb"),
                    "file_name": file_path,
                },
                "strategy": shared.Strategy.HI_RES,
                "languages": ["eng"],
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
                "extract_image_block_types": ["Image"],
            }
        }
        try:
            resp = client.general.partition(request=req)
            output_list = resp.elements
            output_list_modified = tables_from_html(output_list)
            output_list_modified = filter_words(output_list_modified, min_words=20)

            with open(f"convert/output/{file}.json", "w", encoding="utf-8") as f:
                json.dump(output_list_modified, f, ensure_ascii=False, indent=2)
        except SDKError as e:
            print(e)
