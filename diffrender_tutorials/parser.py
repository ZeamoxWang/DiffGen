import pandas as pd
import re
import os

data = {}

def add_data(model_name, epoch, size, loss, wrm, urm):
    key = (model_name, epoch, size)
    value = (loss, wrm, urm)
    data[key] = value

def parse_records(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:

        size_match = re.match(r'^size:(\d+)', line)
        epoch_match = re.match(r'^epoch:(\d+)', line)
        loss_match = re.match(r'^([A-Z_]+)loss:(\S+)$', line)
        weighted_recall_match = re.match(r'^[A-Z_]+weighted order recall:(\S+)', line)
        unweighted_recall_match = re.match(r'^[A-Z_]+unweighted order recall:(\S+)', line)

        if size_match:
            tmp_size = int(size_match.group(1))
        elif epoch_match:
            tmp_epoch = int(epoch_match.group(1))
        elif loss_match:
            tmp_name = loss_match.group(1)
            tmp_loss = float(loss_match.group(2))
        elif weighted_recall_match:
            tmp_wrm = float(weighted_recall_match.group(1))
        elif unweighted_recall_match:
            tmp_urm = float(unweighted_recall_match.group(1))
            # create a new record
            add_data(tmp_name, tmp_epoch, tmp_size, tmp_loss, tmp_wrm, tmp_urm)


# 示例用法
file_path = 'output10-100.txt'
parsed_records = parse_records(file_path)

# 将记录转换为DataFrame
df_data = pd.DataFrame([(key[0], key[1], key[2], value[0], value[1], value[2]) for key, value in data.items()],
                       columns=["Model Name", "Epoch", "Size", "Loss", "weighted_recall", "unweighted_recall"])


# 保存DataFrame为Excel文件
base_name, _ = os.path.splitext(os.path.basename(file_path))
excel_path = f'{base_name}.xlsx'
df_data.to_excel(excel_path, index=True)

print(f"Records successfully converted and saved to {excel_path}.")
