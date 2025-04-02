import os
import json
import pandas as pd
import re  # 用于正则匹配

# 设置根目录路径
root_dir = "/home/zhengtinghua/shenchao/aiops-scwarn/new_res"

# 存储最终的结果
result_list = []

# 遍历所有 service_name 目录
for service_name in os.listdir(root_dir):
    service_path = os.path.join(root_dir, service_name)
    
    if os.path.isdir(service_path):
        # 遍历每个 result_json_and_csv_* 目录
        for folder in os.listdir(service_path):
            folder_path = os.path.join(service_path, folder)
            
            # 判断目录名是否符合 result_json_and_csv_40003 格式
            if folder.startswith("result_json_and_csv_"):
                id_value = folder.split("_")[-1]  # 获取 ID，通常是最后的数字部分
                
                # 获取 JSON 映射文件路径（更深层次的目录结构）
                json_file = os.path.join(folder_path, "result_json", f"result_{id_value}.json")
                
                # 读取 JSON 映射文件
                try:
                    with open(json_file, 'r') as f:
                        json_data = f.read()  # 读取为字符串
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON file {json_file}: {e}")
                    continue  # 跳过无效的 JSON 文件
                except Exception as e:
                    print(f"Unexpected error reading {json_file}: {e}")
                    continue
                
                # 使用正则表达式提取文件名和对应的 KPI 名称
                file_to_kpi = {}
                pattern = r'"([a-f0-9]{32})"\s*:\s*"([^"]+)"'  # 匹配形如 "文件名": "指标名" 的格式

                matches = re.findall(pattern, json_data)  # 查找所有匹配的文件名和指标名
                
                # 将匹配到的文件名和指标名添加到字典
                for file_id, kpi_name in matches:
                    file_to_kpi[file_id] = kpi_name

                # 打印 file_to_kpi 字典，检查映射是否正确
                print(f"File to KPI mapping for id {id_value}: {file_to_kpi}")
                
                # 遍历 CSV 文件并提取信息
                csv_folder_path = os.path.join(folder_path, f"result_csv/{id_value}")
                for csv_file in os.listdir(csv_folder_path):
                    if csv_file.endswith(".csv"):
                        # 排除 _train_origin.csv 结尾的文件
                        if csv_file.endswith("_train_origin.csv"):
                            print(f"Skipping {csv_file} because it ends with '_train_origin.csv'")
                            continue
                        
                        csv_file_path = os.path.join(csv_folder_path, csv_file)
                        
                        # 获取 KPI 名称（从 JSON 映射中查找）
                        file_id = csv_file.split('.')[0]  # 获取文件名（不带扩展名）
                        kpi_name = file_to_kpi.get(file_id, "unknown")  # 如果没有找到，则为 "unknown"
                        
                        # 打印每个文件和对应的 kpi_name
                        print(f"Processing file {csv_file_path} -> KPI: {kpi_name}")
                        
                        # 读取 CSV 文件
                        try:
                            csv_data = pd.read_csv(csv_file_path)
                            
                            # 如果没有 'model_label' 列，跳过该文件
                            if 'model_label' not in csv_data.columns:
                                print(f"Warning: {csv_file_path} does not contain 'model_label' column. Skipping.")
                                continue
                            
                            # 筛选 model_label 为 1 的行
                            filtered_data = csv_data[csv_data['model_label'] == 1]
                            
                            # 提取 id, service_name, kpi_name, timestamp, value
                            for _, row in filtered_data.iterrows():
                                result_list.append({
                                    'id': id_value,
                                    'service_name': service_name,
                                    'kpi_name': kpi_name,
                                    'timestamp': row['timestamp'],
                                    'value': row['origin_value']
                                })
                        except Exception as e:
                            print(f"Error reading CSV file {csv_file_path}: {e}")

# 将结果保存到一个 DataFrame
df_result = pd.DataFrame(result_list)

# 导出结果为 CSV 文件
output_path = "output.csv"
df_result.to_csv(output_path, index=False)

print(f"Result saved to {output_path}")
