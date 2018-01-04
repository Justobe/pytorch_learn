from __future__ import print_function
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader


class MyDataset(data.Dataset):
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

    def __getitem__(self, index):  # 返回的是tensor
        input, target = self.words[index], self.labels[index]
        return input, target

    def __len__(self):
        return len(self.words)

'''
# 生成数据
data_tensor = torch.randn(10)
target_tensor = torch.rand(10)

# 将数据封装成Dataset
tensor_dataset = MyDataset(data_tensor, target_tensor)
#print(data_tensor)
#print(target_tensor)

tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)
for batch in enumerate(tensor_dataloader):
    print("batch\n")
    for data in batch[1]:
        print("data: \n")
        print(data)
'''