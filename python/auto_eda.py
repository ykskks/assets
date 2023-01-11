# pip install --user sweetviz

import numpy as np
import pandas as pd
import sweetviz as sv


class Config:
    input_path = "xxx.csv"
    input_encoding = "utf8"
    output_dir = "xxx"


data = pd.read_csv(Config.input_path, encoding=Config.input_encoding)
data = data.replace([np.inf, -np.inf], [np.nan, -np.nan])  # sweetviz can't handle inf

sv.config_parser.read_string("[General]\nuse_cjk_font=1")
sv = sv.analyze(data)
sv.show_html(f"{Config.output_dir}/sv_report.html")
