from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
from marker.config.parser import ConfigParser

# 这里选择pdf文件作为输入
FILEPATH = "./pdf/Attention Is All You Need.pdf"
# 输出文件夹
OUTPUT_DIR = "./output"
# 相当于给输出文件前面加一个基名
FNAME_BASE = "output"


config = {"output_format": "json"}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
)
rendered = converter(FILEPATH)
text, _, images = text_from_rendered(rendered)
save_output(rendered, OUTPUT_DIR, FNAME_BASE)
