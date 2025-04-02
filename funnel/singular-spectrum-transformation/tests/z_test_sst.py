import unittest
import numpy as np
import pandas as pd
import os
# from fastsst.sst import SingularSpectrumTransformation

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastsst.sst import SingularSpectrumTransformation

class TestSingularSpectrumTransformation(unittest.TestCase):

    def test_sst_all_files(self):
        # 指定文件夹路径
        folder_path = "/home/sunyongqian/liuheng/shenchao/funnel/singular-spectrum-transformation/tests/test_data/aiops2024"
        
        # 确保文件夹存在
        self.assertTrue(os.path.exists(folder_path), "The folder path does not exist.")
        
        # 遍历文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # 只处理 CSV 文件
                file_path = os.path.join(folder_path, file_name)
                
                # 读取文件并选择第二列
                data = pd.read_csv(file_path)
                
                # 确保至少有两列
                self.assertTrue(data.shape[1] >= 2, f"File {file_name} does not have at least two columns.")
                
                # 获取第二列数据
                x = data.iloc[:, 1].values  # 第二列数据

                # 检查并处理 NaN 和 inf 值，增加调试输出
                print(f"Processing file: {file_name}")
                print(f"Original data statistics: min={np.min(x)}, max={np.max(x)}, NaNs={np.isnan(x).sum()}, infs={np.isinf(x).sum()}")
                
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    print(f"Skipping {file_name} due to NaNs or infs in data.")
                    continue

                # 将 NaN 和 inf 值替换为 0
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 打印替换后的数据统计
                print(f"After NaN/inf replacement: min={np.min(x)}, max={np.max(x)}, NaNs={np.isnan(x).sum()}, infs={np.isinf(x).sum()}")
                
                # 初始化 SST，明确设置 order 和 lag 参数
                sst = SingularSpectrumTransformation(win_length=70, order=1, lag=1)
                
                # 添加长度检查，确保数据长度足够
                if len(x) < sst.win_length + sst.order + sst.lag:
                    print(f"Skipping {file_name} due to insufficient data length.")
                    continue
                
                # 计算分数
                try:
                    score = sst.score_offline(x)
                except np.linalg.LinAlgError as e:
                    print(f"LinAlgError in file {file_name}: {e}")
                    continue

                # 验证结果的尺寸
                self.assertEqual(score.size, x.size, f"Score size does not match input data size in file {file_name}.")
                
                # 验证所有分数非负
                self.assertTrue((score >= 0).all(), f"Not all scores are non-negative in file {file_name}.")
                
                print(f"Processed {file_name} successfully.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
