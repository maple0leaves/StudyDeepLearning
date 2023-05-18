import torch

x = torch.Tensor([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]])
print(x.view(1,-1))
print(torch.squeeze(x,dim=0).shape)

if __name__=="__main__":
    # 使用enumerate函数同时迭代元素和它们的索引：
    my_list = ["a", "b", "c"]
    for index, item in enumerate(my_list):
        print(index, item)

        x= torch.Tensor(1,2,3,4)
        print(len(x))