import torch
from geomloss.geomloss import SamplesLoss
from torch.autograd import Function


def label_2_onehot(label, C, device):
    ##transform the InD labels into one-hot vector
    
    size = label.shape[0]
    if len(label.shape) == 1:
        label = torch.unsqueeze(label, 1)
    
    label = label % C
    
    label_onehot = torch.FloatTensor(size, C).to(device)

    label_onehot.zero_()
    label_onehot.scatter_(1, label, 1)
    
    return label_onehot

def custom_cost(X,Y):
    
    if len(X.shape) == 2:
        N,D = X.shape
        M,D = Y.shape
        
        return (1 - torch.eye(N, M)).to('cuda')
    
    if len(X.shape) == 3:
        B,N,D = X.shape
        B,M,D = Y.shape
        
        return torch.unsqueeze(1 - torch.eye(N, M), 0).repeat(B, 1, 1).to('cuda')

def sink_dist_test(input, target, C, device):
    
    test_label_onehot = label_2_onehot(target, C, device)
    ##reshape into (B,N,D)
    test_label_onehot = torch.unsqueeze(test_label_onehot, -1)
    test_input = torch.unsqueeze(input, -1)
    ##Loss value for InD samples
    test_loss = SamplesLoss("sinkhorn", p=2, blur=1.) #Wasserstein-1 distance
    test_loss_value = test_loss(test_input[:,:,0], test_input, test_label_onehot[:,:,0], test_label_onehot)
    
    return test_loss_value


def sink_dist_test_v2(input, target, C, device):
    
    all_class = torch.LongTensor([i for i in range(C)]).to(device)
    all_class_onehot = label_2_onehot(all_class, C, device)
    ##reshape into (B,N,D)
    all_class_onehot = torch.unsqueeze(all_class_onehot, -1)
    test_input = torch.unsqueeze(input, -1)
    test_batch_size = test_input.shape[0]
    test_loss_values = torch.zeros(test_batch_size, C).to(device)
    test_loss = SamplesLoss("sinkhorn", p=2, blur=1., cost = custom_cost) #Wasserstein-1 distance
    for b in range(test_batch_size):
        input_b = test_input[b:b+1,:,:].repeat(C, 1, 1)
        test_loss_values[b] = test_loss(input_b[:,:,0], input_b, all_class_onehot[:,:,0], all_class_onehot)
    
    return test_loss_values.min(dim=1)[0]

    
class NLLWOOD_Loss_v2(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    
    def forward(ctx, Input, Target, C, beta, device):
        """
        input: (N,C), N is the batch size, C is the number of Class
        target: (N), 0,...,C-1, for in distribution, C for out of distribution, the data type should be int
        C: the number of class
        """
        input = Input.clone()
        target = Target.clone()
        #print(input.requires_grad)
        
        ##find the OOD and InD samples in training batch
        OOD_ind = (target == C).nonzero(as_tuple=True)[0]
        #print(OOD_ind)
        OOD_input = input[OOD_ind]
        
        InD_ind = (target != C).nonzero(as_tuple=True)[0]
        InD_input = input[InD_ind]
        #print(InD_ind)
        InD_label = target[InD_ind] ##only InD samples have labels
        all_class = torch.LongTensor([i for i in range(C)]).to(device)
        
        ##transform the InD labels into one-hot vector
        InD_label_onehot = label_2_onehot(InD_label, C, device)
        
        ##Loss value for InD samples
        log_input = InD_input.log()
        InD_loss = torch.nn.NLLLoss()
        InD_loss_value = InD_loss(log_input, InD_label)
        
        
        ##Loss value for OOD samples
        all_class_onehot = label_2_onehot(all_class, C, device)
        all_class_onehot = torch.unsqueeze(all_class_onehot, -1)
        
        OOD_loss = SamplesLoss("sinkhorn", p=2, blur=1., cost = custom_cost)
        OOD_input = torch.unsqueeze(OOD_input, -1)
        OOD_batch_size = OOD_input.shape[0]
        #print(OOD_input)
        
        
        ####elminate min in label####
        OOD_loss_values = torch.zeros(OOD_batch_size, C).to(device)
        for b in range(OOD_batch_size):
            input_b = OOD_input[b:b+1,:,:].repeat(C, 1, 1)
            OOD_loss_values[b] = OOD_loss(input_b[:,:,0], input_b, all_class_onehot[:,:,0], all_class_onehot)
        
        values, idx = torch.min(OOD_loss_values, dim=1)
        

        ctx.save_for_backward(InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C, idx)
        
        loss_value = InD_loss_value - beta * torch.mean(values)
        
        
        
        
        return loss_value

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C, min_idx_OOD, min_idx_all = ctx.saved_tensors
        InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C, min_ind= ctx.saved_tensors
        
        InD_batch_size = InD_input.shape[0]
        
        
        OOD_loss = SamplesLoss("sinkhorn", p=2, blur=1., potentials=True, cost = custom_cost)
        OOD_batch_size = OOD_input.shape[0]
        
        
        OOD_f = torch.zeros(OOD_batch_size, C).to('cuda')
        for b in range(OOD_batch_size):
            input_b = OOD_input[b:b+1,:,:].repeat(C, 1, 1)
            
            f, _ = OOD_loss(input_b[:,:,0], input_b, all_class_onehot[:,:,0], all_class_onehot)
            
            OOD_f[b] = f[min_ind[b]]
        
        
        
        #print(OOD_ind, InD_ind)
        grad_Input = torch.zeros([InD_batch_size+OOD_batch_size, C]).to('cuda')
        
        
        grad_Input[OOD_ind,:] = -beta * OOD_f
        grad_Input[InD_ind,:] = -InD_label_onehot * (1. / InD_batch_size)
        
        
        return grad_Input, None, None, None, None
    
    
class NLLWOOD_Loss(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    
    def forward(ctx, Input, Target, C, beta, device):
        """
        input: (N,C), N is the batch size, C is the number of Class
        target: (N), 0,...,C-1, for in distribution, C for out of distribution, the data type should be int
        C: the number of class
        """
        input = Input.clone()
        target = Target.clone()
        #print(input.requires_grad)
        
        ##find the OOD and InD samples in training batch
        OOD_ind = (target == C).nonzero(as_tuple=True)[0]
        #print(OOD_ind)
        OOD_input = input[OOD_ind]
        
        InD_ind = (target != C).nonzero(as_tuple=True)[0]
        InD_input = input[InD_ind]
        #print(InD_ind)
        InD_label = target[InD_ind] ##only InD samples have labels
        all_class = torch.LongTensor([i for i in range(1)]).to(device)
        
        ##transform the InD labels into one-hot vector
        InD_label_onehot = label_2_onehot(InD_label, C, device)
        
         ##Loss value for InD samples
        log_input = InD_input.log()
        InD_loss = torch.nn.NLLLoss()
        InD_loss_value = InD_loss(log_input, InD_label)
       
        
        ##Loss value for OOD samples
        all_class_onehot = label_2_onehot(all_class, C, device)
        all_class_onehot = torch.unsqueeze(all_class_onehot, -1)
        OOD_loss = SamplesLoss("sinkhorn", p=2, blur=1.)
        OOD_input = torch.unsqueeze(OOD_input, -1)
        OOD_batch_size = OOD_input.shape[0]
        #print(OOD_input)
        
        
        ####elminate min in label####
        all_class_onehot = all_class_onehot.repeat(OOD_batch_size,1,1)
        OOD_loss_value = OOD_loss(OOD_input[:,:,0], OOD_input, all_class_onehot[:,:,0], all_class_onehot).mean()
        ctx.save_for_backward(InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C)
        ####
        
        
        
        loss_value = InD_loss_value - beta * OOD_loss_value
        
        
        
        
        return loss_value

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C, min_idx_OOD, min_idx_all = ctx.saved_tensors
        InD_input, InD_label_onehot, OOD_input, all_class_onehot, beta, OOD_ind, InD_ind, C= ctx.saved_tensors
        
        InD_batch_size = InD_input.shape[0]
        
        
        OOD_loss = SamplesLoss("sinkhorn", p=2, blur=1., potentials=True)
        OOD_batch_size = OOD_input.shape[0]
        OOD_f, OOD_g = OOD_loss(OOD_input[:,:,0], OOD_input, 
                                all_class_onehot[0:1].repeat(OOD_batch_size,1,1)[:,:,0],
                                all_class_onehot[0:1].repeat(OOD_batch_size,1,1))
        
        
        
        #print(OOD_ind, InD_ind)
        grad_Input = torch.zeros([InD_batch_size+OOD_batch_size, C]).to('cuda')
        
        
        grad_Input[OOD_ind,:] = -beta * OOD_f
        grad_Input[InD_ind,:] = -InD_label_onehot * (1. / InD_batch_size)
        
        
        return grad_Input, None, None, None, None
