import os
import shutil

# 源路径
source_dir = '/home/zhengtinghua/shenchao/aiops-scwarn/new_res'
# 目标路径
destination_dir = '/home/zhengtinghua/shenchao/aiops-scwarn/new_new_res'  # 需要指定目标路径
# 新的文件夹命名起始值
new_folder_name_prefix = 'result_json_and_csv_'
new_folder_number = 70001  # 从70001开始

def copy_result_folders():
    global new_folder_number
    # 遍历源路径下的所有文件夹
    for root, dirs, files in os.walk(source_dir):
        # 找到包含 'result_json_and_csv' 的文件夹
        for folder in dirs:
            # 判断当前文件夹是否为结果文件夹
            if 'result_json_and_csv' in folder:
                source_folder = os.path.join(root, folder)
                # 动态生成新的文件夹名称，确保编号格式为五位数
                new_folder_name = f"{new_folder_name_prefix}{new_folder_number}"
                new_folder_path = os.path.join(destination_dir, new_folder_name)
                
                # 复制文件夹及其内容
                try:
                    shutil.copytree(source_folder, new_folder_path)
                    print(f"Copied: {source_folder} -> {new_folder_path}")
                    new_folder_number += 1  # 更新文件夹编号
                except Exception as e:
                    print(f"Error copying {source_folder}: {e}")

if __name__ == '__main__':
    copy_result_folders()
