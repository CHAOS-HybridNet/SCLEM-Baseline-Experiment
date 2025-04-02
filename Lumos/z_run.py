import os
import pandas as pd
import hashlib
import json
from mct.BiasTester import BiasTester
from mct.MetricComparer import MetricComparer

# 生成字符串的MD5码
def generate_md5(input_string):
    md5_object = hashlib.md5()
    md5_object.update(input_string.encode('utf-8'))
    return md5_object.hexdigest()

# 处理单个 ID 的函数
def process_id(id):
    # 创建结果保存的文件夹
    results_dir = f'results_{id}'
    os.makedirs(results_dir, exist_ok=True)

    # 定义训练文件路径
    train_kpi_file = f'/home/zhengtinghua/shenchao/baseline/new_dataset/train/{id}/train_kpi.csv'
    train_log_file = f'/home/zhengtinghua/shenchao/baseline/new_dataset/train/{id}/train_log.csv'
    train_output_file = f'/home/zhengtinghua/shenchao/baseline/new_dataset/train/{id}/train_combined.csv'
    
    # 创建文件用于保存列名和MD5码
    column_names_file = os.path.join(results_dir, 'column_names_md5.txt')

    # 读取训练 CSV 文件并合并
    kpi_df = pd.read_csv(train_kpi_file)
    log_df = pd.read_csv(train_log_file)
    if len(kpi_df) != len(log_df):
        raise ValueError(f"训练数据的行数不一致 (ID: {id})")

    combined_df = pd.concat([log_df, kpi_df], axis=1)
    md5_column_names = {col: generate_md5(col) for col in combined_df.columns}
    
    # 将列名和MD5码保存到文件
    with open(column_names_file, 'w') as f:
        for original, md5_name in md5_column_names.items():
            f.write(f"Original column name: {original}, MD5 hash: {md5_name}\n")
            print(f"Original column name: {original}, MD5 hash: {md5_name}")

    combined_df.rename(columns={combined_df.columns[-1]: 'target_metric'}, inplace=True)
    combined_df.to_csv(train_output_file, index=False)

    # 保存到目标目录
    target_dir = f'/home/zhengtinghua/shenchao/baseline/Lumos/results_{id}'
    os.makedirs(target_dir, exist_ok=True)
    output_file = os.path.join(target_dir, 'train_combined.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"训练数据已成功合并并保存到 {output_file}")

    # 定义测试文件路径
    test_kpi_file = f'/home/zhengtinghua/shenchao/baseline/new_dataset/test/{id}/test_kpi.csv'
    test_log_file = f'/home/zhengtinghua/shenchao/baseline/new_dataset/test/{id}/test_log.csv'
    test_output_file = f'/home/zhengtinghua/shenchao/baseline/new_dataset/test/{id}/test_combined.csv'

    # 读取测试 CSV 文件并合并
    kpi_df = pd.read_csv(test_kpi_file)
    log_df = pd.read_csv(test_log_file)
    if len(kpi_df) != len(log_df):
        raise ValueError(f"测试数据的行数不一致 (ID: {id})")

    combined_df = pd.concat([log_df, kpi_df], axis=1)
    md5_column_names = {col: generate_md5(col) for col in combined_df.columns}
    
    # 将测试文件的列名和MD5码也保存到同一文件
    with open(column_names_file, 'a') as f:
        for original, md5_name in md5_column_names.items():
            f.write(f"Original column name: {original}, MD5 hash: {md5_name}\n")
            print(f"Original column name: {original}, MD5 hash: {md5_name}")

    combined_df.rename(columns={combined_df.columns[-1]: 'target_metric'}, inplace=True)
    combined_df.to_csv(test_output_file, index=False)

    # 保存到目标目录
    output_file = os.path.join(target_dir, 'test_combined.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"测试数据已成功合并并保存到 {output_file}")

    # 进行比较和偏差检测
    control = pd.read_csv(train_output_file, na_values=["", "nan", "NaN", "#NULL#", "#NUL#"])
    treatment = pd.read_csv(test_output_file, na_values=["", "nan", "NaN", "#NULL#", "#NUL#"])
    config_file = 'config_new.json'
    with open(config_file) as file:
        config = json.load(file)

    delta_comparer = MetricComparer(config)
    metric_delta = delta_comparer.compare(control, treatment)
    
    bias_tester = BiasTester(config)
    bias_results, deviation, is_biased = bias_tester.check_bias(control, treatment)
    n_control, n_treatment = bias_tester.normalize_bias(control, treatment, bias_results)
    n_bias_results, n_deviation, n_is_biased = bias_tester.check_bias(n_control, n_treatment)
    n_metric_delta = delta_comparer.compare(n_control, n_treatment)

    # 保存结果到文件夹
    metric_delta.to_csv(os.path.join(results_dir, 'metric_delta.csv'), index=False)
    bias_results.to_csv(os.path.join(results_dir, 'bias_results.csv'), index=False)
    n_bias_results.to_csv(os.path.join(results_dir, 'n_bias_results.csv'), index=False)
    n_metric_delta.to_csv(os.path.join(results_dir, 'n_metric_delta.csv'), index=False)

# 遍历 ID 范围 70001 到 70050
for id in range(70001, 70051):
    try:
        process_id(str(id))
    except Exception as e:
        print(f"处理 ID {id} 时出现错误: {e}")
