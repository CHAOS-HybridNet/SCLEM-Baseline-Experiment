import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 获取文件夹中所有 CSV 文件路径
folder_path = '/home/zhengtinghua/shenchao/baseline/funnel/singular-spectrum-transformation/tests/test_data/aiops2024'
csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# 2. 遍历每个 CSV 文件，并处理数据
for file in csv_files:
    try:
        # 读取数据，假设文件包含 'timestamp' 和 'value' 两列
        df = pd.read_csv(file, header=0, names=['timestamp', 'value'])

        # 确保数据正常加载
        if df.empty:
            print(f"{file} 是空的，跳过此文件。")
            continue

        # 检查数据格式
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            print(f"{file} 缺少必要的列，跳过此文件。")
            continue

        # 打印文件名和数据概况
        print(f"正在处理文件：{file}")
        print(df.head())  # 打印前几行数据进行调试

        # 将时间戳转换为顺序数值（可以按实际需要直接使用时间戳，或者转换为时间序列）
        df['year'] = pd.to_numeric(df['timestamp'], errors='coerce').rank().astype(int)

        # 设定处理前后的时间区间（根据数据的长度设置）
        PRE_TEREM = [1, len(df) // 2]  # 前一半时间作为 PRE_TEREM
        POST_TEREM = [len(df) // 2 + 1, len(df)]  # 后一半时间作为 POST_TEREM

        # 打印区间信息用于调试
        print(f"PRE_TEREM: {PRE_TEREM}, POST_TEREM: {POST_TEREM}")

        # 绘制图像
        plt.figure()
        plt.plot(df['year'], df['value'], label='Value Over Time')
        plt.title(f"Time Series Plot for {os.path.basename(file)}")
        plt.xlabel("Time (Rank of Timestamps)")
        plt.ylabel("Value")
        plt.legend()

        # 指定保存图像的文件夹
        output_folder = '/home/zhengtinghua/shenchao/baseline/funnel/pysynthdid/output_plots_2'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存图像到指定文件夹
        plot_filename = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_plot.png'))
        plt.savefig(plot_filename)
        plt.close()  # 关闭当前绘图窗口，继续处理下一个文件

        print(f"图像已成功保存：{plot_filename}")

    except Exception as e:
        print(f"处理文件 {file} 时出错：{str(e)}，跳过此文件。")
