import torch

if __name__ == "__main__":
    input = torch.tensor([[1, 2 ,4 ,7], [2, 1 ,12 ,14], [12, 1 ,2 ,16], [7, 6 ,1 ,13]])
    min  = torch.tensor([7, 1 , 2 , 4])
    mask = torch.le(input, min)
    print(input)
    print(mask)
    final = torch.max(mask * input, dim=0)
    print(mask * input)
    print(final)
