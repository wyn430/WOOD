import torch



def label_2_onehot(label, C):
    ##transform the InD labels into one-hot vector
    size = label.shape[0]
    if len(label.shape) == 1:
        label = torch.unsqueeze(label, 1)

    label = torch.LongTensor(label) % C

    label_onehot = torch.FloatTensor(size, C)

    label_onehot.zero_()
    label_onehot.scatter_(1, label, 1)
    
    return label_onehot
        
       