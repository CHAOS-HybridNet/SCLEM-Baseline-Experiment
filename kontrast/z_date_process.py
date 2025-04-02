import pandas as pd
import os


name="bkverify/"
id="29867"


def split_data(file_path, output_dir):
    # 读取源文件
    # print(file_path)
    df = pd.read_csv(file_path)

    # 创建输出目录，如果不存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历除了第一列（timestamp）的每一列
    for i, col in enumerate(df.columns[1:], start=0):
        # 创建新表格，只包含timestamp和当前列
        new_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'value': df[col]
        })
        
        output_file = os.path.join(output_dir, f'none$inst_{id}_{i}.csv')
        
        # 将新表格保存为Excel文件
        new_df.to_csv(output_file, index=False)
        print(f"已保存: {output_file}")




# 使用示例
file_path = f'/home/sunyongqian/liuheng/aiops-scwarn/data/daily/{name+id}/test_kpi.csv'  # 替换为你的源文件路径
output_dir = '/home/sunyongqian/liuheng/shenchao/kontrast/dataset/data/aiops2024'       # 替换为输出目录路径

split_data(file_path, output_dir)
