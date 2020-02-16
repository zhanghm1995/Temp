import numpy as np

dic = {1:"zhanghaiming", 2:"css", 3:"zhangjiacheng"}

remove_list = [1,3]

print(dic)
for id in remove_list:
    del dic[id]

print(dic)

number = np.array([[1,2], [1,3]])

print("shape is {}, number is {}".format(number.shape, number))