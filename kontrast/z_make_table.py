import os
import pandas as pd

def create_table_from_files(folder_path):
    # 初始化空列表用于存储表格数据
    table_data = []

    # 遍历文件夹中的所有CSV文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            # 去除文件名中的后缀
            metric_name = file_name[:-4]
            
            # 删除"none$"前缀
            metric_name = metric_name.replace('none$', '')
            
            # 读取CSV文件
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            
            # 获取第一列的第一个和最后一个数据
            change_start = df.iloc[0, 0]
            change_end = df.iloc[-1, 0]
            
            # 解析文件名中的 i 值
            file_parts = metric_name.split('_')
            if len(file_parts) > 1:
                i_value = int(file_parts[1])
            else:
                i_value = 0  # 如果文件名格式不正确，默认i值为0
            
            # 计算label和case_label
            label = 0 if i_value >= 11000 else 1
            
            # 将数据添加到列表
            table_data.append({
                'instance_name': 'none',
                'metric': metric_name,
                'change_start': change_start,
                'change_end': change_end,
                'label': label,
                'case_id': i_value,
                'case_label': label
            })
    
    # 将列表转化为DataFrame
    df_result = pd.DataFrame(table_data)

    # 按第二列 'metric' 进行排序
    df_result = df_result.sort_values(by='metric')

    # 输出表格
    return df_result

# 使用示例
folder_path = '/home/zhengtinghua/shenchao/baseline/kontrast/dataset/data/aiops2024_2'  # 替换为包含CSV文件的文件夹路径
df_result = create_table_from_files(folder_path)

# 将结果保存为CSV或打印输出
df_result.to_csv('output_table.csv', index=False)
print(df_result)
