import random
random.seed(2022)
import os


def data_k_split(vcug_dir, k=5):
    """
    将数据集按照k份进行划分，然后将划分的结果写入到txt文件中
    """
    vcug_jsons = [file for file in os.listdir(vcug_dir)]
    vcug_jsons_splits = k_split(vcug_jsons, k=k, shuffle=True)
    for i in range(len(vcug_jsons_splits)):
        test = vcug_jsons_splits[i]
        train = []
        for j in range(k):
            if j==i:
                continue
            else:
                train=train+vcug_jsons_splits[j]
        with open(os.path.join(vcug_dir, 'train_'+str(i)+'.txt'), 'w') as f:
            for t in train:
                f.write(t)
                f.write('\n')
        with open(os.path.join(vcug_dir, 'test_'+str(i)+'.txt'), 'w') as f:
            for t in test:
                f.write(t)
                f.write('\n')


def k_split(full_list, k, shuffle=False):
    """
    将full_list按照k折进行拆分，返回一个[[],[],[],[],]形式的列表
    k即k折
    shuffle是随机的意思
    """
    n_total = len(full_list)
    offset = n_total // k
    if n_total == 0 or offset < 1:
        return []
    if shuffle:
        random.shuffle(full_list)
    split_result = []
    for i in range(k):
        split_result.append(full_list[i*offset:(i+1)*offset])
    return split_result


def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1 列表*ratio
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

if __name__ == '__main__':
    data_dir = '/home/tanzl/data/VCUG/trainTest/doubleBranch/'
    data_k_split(vcug_dir=data_dir)
