'''
@version: 0.0.1
@author: yuhuan
@Contact: redaihanyu@126.com
@site: 
@file: q_2.py
@time: 2017/12/2 下午8:03

'''

'''
【问题】
给你一个很大的文本文件，无法全部读到内存。
要求：
a.现在需要从里面随机采样（每行被选出的概率要一样）出1000行进行检查，请实现出一个函数从而实现这个功能
b.要求尽可能的高效

-------------------------------------------------------------

【思路】
问题简化为两个小问题：
    1、假设文本文件有 n 行，需要从【 0 ~ n - 1 】这个数列中随机找出不重复的 m 个数字(m <= n)，每个数字取出的概率相同；
    2、将大文件拆分成许多小的文件，将小文件读入内存中。
'''
import random
import os

# 从【0 ~ n - 1】这个数列中等概率随机抽取 m 个数字， m <= n。
def get_row_num(n, m):
    if m > n or n <= 0 or m <= 0:
        print("Error, must 0 < m <= n")
        exit(0)

    result  = []
    select = m
    remaining = n
    for i in range(n):
        if random.randint(0, n) % remaining < select:
            result.append(i)
            select -= 1
        remaining -= 1

    return result


# 文件分割并选择
def split(fromfile, todir, chunksize):
    # 检查目标目录是否存在
    if not os.path.exists(todir):
        os.mkdir(todir)
    else:
        for fname in os.listdir(todir):
            os.remove(os.path.join(todir, fname))

    partnum = 0
    inputfile = open(fromfile, 'rb')
    while True:
        chunk = inputfile.read(chunksize)
        # 检查是否为空
        if not chunk:
            break
        partnum += 1
        filename = os.path.join(todir, ('part%04d'%partnum))
        # 写入分割
        fileobj = open(filename, 'wb')
        fileobj.write(chunk)
        fileobj.close()

# 按照行数，读取一行样本
def get_samples(file_path, rows_num):
    # 分割大文件
    todir = 'tempdir'
    split(file_path, todir, 400000000)  # 400000000bit 大概380M

    read_count = 0  # 记录已经读取了的行数
    # 循环读取小文件
    for filename in os.listdir(todir):
        for line in open(filename):
            # 如果是我们将要选择的那一行样本，输出
            if read_count == rows_num[0]:
                print(line)     # 输出结果
                rows_num.pop(0)  # 已经读取的就没必要检查了
            read_count += 1


# 前提是文本文件太大，无法读入内存
if __name__ == '__main__':
    # 假设文本文件有1000000行，(Linux环境下，wc -l 就可以得到文本文件的行数)
    rows = get_row_num(1000000, 1000)
    get_samples("bigfile_path", rows)   # 选择样本



