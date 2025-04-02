import os
import subprocess

# 根目录
root_dir = '/home/sunyongqian/liuheng/weihua/data_20241019/data_as_SCWarn/test'

# 其他固定参数
publish_date = '20240620'
prometheus_address = '172.16.17.114:19192'
train_date = '2024-10-16 18:29:13'
task_count = 5
step = 120
timeout = 10000
train_duration = 691200
detection_duration = 86400
predict_interval = '30'

# 遍历根目录下所有文件夹
for service in os.listdir(root_dir):
    service_path = os.path.join(root_dir, service)
    
    # 仅处理文件夹
    if os.path.isdir(service_path):
        # 遍历一级子文件夹，获取 sc_id
        for sc_id in os.listdir(service_path):
            sc_id_path = os.path.join(service_path, sc_id)
            
            # 确保 sc_id 是文件夹
            if os.path.isdir(sc_id_path):
                # 构建命令
                command = (
                    f"python3 -m run "
                    f"--publish_date {publish_date} "
                    f"--prometheus_address {prometheus_address} "
                    f"--service {service} "
                    f"--sc_id {sc_id} "
                    f"--train_date '{train_date}' "
                    f"--task_count {task_count} "
                    f"--step {step} "
                    f"--timeout {timeout} "
                    f"--train_duration {train_duration} "
                    f"--detection_duration {detection_duration} "
                    f"--predict_interval {predict_interval}"
                )
                
                # 打印当前运行的命令
                print(f"\nRunning command for service: {service} with sc_id: {sc_id}")
                
                try:
                    # 运行命令
                    subprocess.run(command, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running command: {e}")
        
        # 等待用户输入字符以继续
        input("\n运行完该 service，按任意键继续运行下一个 service...")

