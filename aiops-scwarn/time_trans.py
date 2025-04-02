import os
import re
import pandas as pd
from datetime import datetime

# 定义主要路径
new_res_path = "new_res"
data_source_path = "data_20241019/data_as_SCWarn/new_test"

# 定义要删除的字符串内容和多余的逗号模式
string_to_remove = '{"d7e6d55ba379a13d08c25d15faf2a23b": "timestamp"}'
comma_pattern = '"dim_info": [, {"'

# 定义正则表达式来匹配 time.struct_time 格式
timestamp_pattern = re.compile(
    r"time\.struct_time\(tm_year=(\d+), tm_mon=(\d+), tm_mday=(\d+), tm_hour=(\d+), tm_min=(\d+), tm_sec=(\d+)"
)

# 遍历new_res文件夹的每个服务文件夹
for service_name in os.listdir(new_res_path):
    service_path = os.path.join(new_res_path, service_name)
    if os.path.isdir(service_path):
        # 遍历每个result_json_and_csv_xxxx子文件夹
        for folder_name in os.listdir(service_path):
            if folder_name.startswith("result_json_and_csv_"):
                result_folder_path = os.path.join(service_path, folder_name)
                
                # 递归删除以d7e开头的文件
                for root, _, files in os.walk(result_folder_path):
                    for file_name in files:
                        if file_name.startswith("d7e"):
                            file_path = os.path.join(root, file_name)
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                
                # 清理JSON文件中的指定字符串和多余逗号
                for root, _, files in os.walk(result_folder_path):
                    for file_name in files:
                        if file_name.endswith(".json"):
                            json_file_path = os.path.join(root, file_name)
                            with open(json_file_path, "r") as f:
                                content = f.read()
                            # 删除指定的字符串
                            if string_to_remove in content:
                                content = content.replace(string_to_remove, "")
                                print(f"Removed specified string from JSON file: {json_file_path}")
                            
                            # 删除 "dim_info": [, {...} 中的多余逗号
                            if comma_pattern in content:
                                content = content.replace('"dim_info": [, {"', '"dim_info": [ {"')
                                print(f"Removed extra comma from JSON file: {json_file_path}")
                                
                            # 写回修改后的内容
                            with open(json_file_path, "w") as f:
                                f.write(content)

                # 处理CSV文件中的 timestamp 列
                for root, _, files in os.walk(result_folder_path):
                    for file_name in files:
                        if file_name.endswith(".csv"):
                            csv_file_path = os.path.join(root, file_name)
                            df = pd.read_csv(csv_file_path)
                            
                            # 检查 timestamp 列并进行格式转换
                            if 'timestamp' in df.columns:
                                def convert_timestamp(value):
                                    match = timestamp_pattern.search(str(value))
                                    if match:
                                        # 提取匹配的时间部分
                                        year, month, day, hour, minute, second = map(int, match.groups())
                                        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
                                    return value  # 返回原始值以防格式不匹配

                                # 应用转换函数
                                df['timestamp'] = df['timestamp'].apply(convert_timestamp)
                                df.to_csv(csv_file_path, index=False)
                                print(f"Updated timestamp format in CSV file: {csv_file_path}")

                # 找到所有train_origin.csv文件
                result_csv_path = os.path.join(result_folder_path, "result_csv", folder_name.split("_")[-1])
                if os.path.exists(result_csv_path):
                    for file_name in os.listdir(result_csv_path):
                        if file_name.endswith("train_origin.csv"):
                            train_origin_path = os.path.join(result_csv_path, file_name)
                            
                            # 定义data_source路径中的test_kpi.csv和test_log.csv文件路径
                            test_kpi_path = os.path.join(data_source_path, service_name, folder_name.split("_")[-1], "test_kpi.csv")
                            test_log_path = os.path.join(data_source_path, service_name, folder_name.split("_")[-1], "test_log.csv")
                            
                            # 根据文件名前缀选择不同的数据源
                            if file_name.startswith("b5") and os.path.exists(test_log_path):
                                # 从test_log.csv读取前两列，保留所有行
                                df = pd.read_csv(test_log_path, usecols=[0, 1])
                                df.columns = [df.columns[0], "origin_value"]
                                df.to_csv(train_origin_path, index=False)
                                print(f"Updated with test_log.csv data: {train_origin_path}")
                            
                            elif file_name.startswith("d1") and os.path.exists(test_kpi_path):
                                # 从test_kpi.csv读取第一列和第三列，保留所有行
                                df = pd.read_csv(test_kpi_path, usecols=[0, 2])
                                df.columns = [df.columns[0], "origin_value"]
                                df.to_csv(train_origin_path, index=False)
                                print(f"Updated with specific columns from test_kpi.csv data: {train_origin_path}")
                            
                            else:
                                # 默认情况，从test_kpi.csv读取前两列，保留所有行
                                if os.path.exists(test_kpi_path):
                                    df = pd.read_csv(test_kpi_path, usecols=[0, 1])
                                    df.columns = [df.columns[0], "origin_value"]
                                    df.to_csv(train_origin_path, index=False)
                                    print(f"Updated with default test_kpi.csv data: {train_origin_path}")
                                else:
                                    print(f"File not found: {test_kpi_path}")
