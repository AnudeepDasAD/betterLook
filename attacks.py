import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import models
import matplotlib.pyplot as plt
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#############FGSM###########################################################
def fgsm(net, x, t, eps=0.01, targ=False):
    '''
        x_adv = FGSM(net, x, t, eps=0.01, targ=False)
        
        Performs the Fast Gradient Sign Method, perturbing each input by
        eps (in infinity norm) in an attempt to have it misclassified.
        
        Inputs:
          net    PyTorch Module object
          x      (D,I) tensor containing a batch of D inputs
          t      tensor of D corresponding class indices
          eps    the maximum infinity-norm perturbation from the input
          targ   Boolean, indicating if the FGSM is targetted
                   - if targ is False, then t is considered to be the true
                     class of the input, and FGSM will work to increase the cost
                     for that target
                   - if targ is True, then t is considered to be the target
                     class for the perturbation, and FGSM will work to decrease the
                     cost of the output for that target class
        
        Output:
          x_adv  tensor of a batch of adversarial inputs, the same size as x
    '''

    # You probably want to create a copy of x so you can work with it.
    x_adv = x.clone().to(device)
    
    # Forward pass
    y = net(x)
    
    # loss
    # loss = net.loss_fcn(y,t)
    loss = torch.nn.CrossEntropyLoss()(y,t)
    net.zero_grad()
    loss.backward()
    
    # The gradients should be populated now because we did backward()
    
    # If targ == True, then we do gradient descent
    multiplier = 1
    if targ:
        multiplier = -1
    
    x_adv = x + multiplier*eps*torch.sign(x.grad)
    return x_adv




##########################################PGD##################################

def pgd(net, x, t, num_steps=40, step_size=4/255, eps=16/255, targ=False, random_start=False):
    x = x.clone().detach().to(device)
    t = t.clone().detach().to(device)
    
    loss = nn.CrossEntropyLoss()

    x_advs = x.clone().detach()

    # Choose a random point in the epsilon-ball around x
    if random_start:
        x_advs = x_advs + torch.empty_like(x_advs).uniform_(-eps, eps)
        x_advs = torch.clamp(x_advs, min=0, max=1).detach()
    
    # Iteratively accumulate the gradient
    for i in range(num_steps):
        x_advs.requires_grad=True
        y = net(x_advs)
        cost = loss(y,t)
        # net.zero_grad()
        # loss.backward()
        grad = torch.autograd.grad(cost, x_advs, retain_graph=False, create_graph=False)[0]
        
        x_advs = x_advs.detach() + step_size*grad.sign()
        delta = torch.clamp(x_advs - x, min=-eps, max=eps)
        x_advs = torch.clamp(x + delta, min=0, max=1).detach()
        
        
    # multiplier = 1 if targ else -1

    # with torch.no_grad():
    #   x_advs += multiplier * grad
    
    # Project x_advs back into the epsilon ball
    
    # x_advs could have gone too high or too far right (more than +eps too far)
    #   and it could have gone too far down or too far left (more than -eps too far) (in the case of 2D)
     
    # We slam all the values in the delta (x_advs-x) so the delta is at most epsilon and at least -epsilon for each dimension
    # delta = torch.clamp(x_advs - x, min=-eps, max=eps)
    
    # revise x_advs to add the delta instead. 
    # x_advs = torch.clamp(x + delta, min=0, max=1)
    
    return x_advs
        



#################################CW###################################################### 
def cw_attack(model, images, labels, c=1, kappa=0, steps=1000, lr=0.01, targeted=False): # VERIFY LR
        r"""
        Overridden.
        """
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        # if targeted:
        #     target_labels = _get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10*torch.ones((len(images))).to(device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = torch.optim.Adam([w], lr=lr)

        for step in range(steps):
            # Get adversarial images
            adv_images = tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            # if targeted:
            #     f_loss = f(outputs, target_labels).sum()
            # else:
            f_loss = f(outputs, labels, kappa).sum()

            cost = L2_loss + c*f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(steps//10,1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

def tanh_space(x):
    return 1/2*(torch.tanh(x) + 1)

def inverse_tanh_space(x):
    # torch.atanh is only for torch >= 1.7.0
    return atanh(x*2-1)

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
def f(outputs, labels, kappa, targeted=False):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

    if targeted:
        return torch.clamp((i-j), min=-kappa)
    else:
        return torch.clamp((j-i), min=-kappa)     





#####################DEEPFOOL################################################

def deepfool(model, images, labels, steps=50, overshoot=0.02, _supported_mode = ['default'], return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        target_labels = labels.clone().detach().to(device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = _forward_indiv(model, adv_images[idx], labels[idx], overshoot)
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    
def _forward_indiv(model, image, label, overshoot):
        image.requires_grad = True
        fs = model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = _construct_jacobian(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label else hat_L+1

        adv_image = image + (1+overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
def _construct_jacobian(y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)


def draw(x):
    '''Displays a flattened MNIST digit'''
    img_size = 32
    with torch.no_grad():
        plt.imshow(x.cpu().numpy().reshape((img_size,img_size)), cmap='gray')
        plt.axis('off')


attacks = {'fgsm':fgsm, 'pgd':pgd, 'cw':cw_attack, 'deepfool':deepfool}

attack = 'pgd'
MODEL_TO_USE = 'resnet'
USE_CONFIG = False

if __name__=='__main__':

    if MODEL_TO_USE == 'resnet':
        weights=models.ResNet50_Weights.IMAGENET1K_V2
        base_model = models.resnet50(weights=weights)
        base_model.eval()
    elif MODEL_TO_USE == 'inception':
        base_model = timm.create_model('inception_v3', pretrained=True)
        base_model.eval()
    elif MODEL_TO_USE == 'incres':
        base_model = timm.create_model('inception_resnet_v2', pretrained=True)
        base_model.eval()
    elif MODEL_TO_USE == 'ens_adv':
        base_model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        base_model.eval()
    else: # '' case
        # Turns out that the performance is shockingly good (~20%) when testing the below model
        #    on imagenet100 (100 classes)
        base_model = torch.load('resnet_glance_trial1_epoch4.pt')


    if MODEL_TO_USE != 'resnet' and MODEL_TO_USE != '':
        config = resolve_data_config({}, model=base_model)
        transform = create_transform(**config)
        USE_CONFIG = True
    
    ds = torch.load('imagenette_preproc_valset.pt')
    x_advs = []
    ts = []
    if not USE_CONFIG:
        dl_test = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
        num_batches_processed = 0
        for x,t in dl_test:
            x = x.to(device)
            t = t.to(device)
            x.requires_grad=True
            x_adv = attacks[attack](base_model.cuda(), x, t, eps = 16).to('cpu') # May also try 32
            # plt.imshow(torch.einsum('ijk->jki', x_advs[0].detach().cpu()), cmap='gray')
            # plt.show()
            x_advs.append(x_adv)
            ts.append(t.to('cpu'))
            num_batches_processed += 1
            print(f'num batches processed: {num_batches_processed}')
            if num_batches_processed % 80 == 0 or num_batches_processed == 491: #625 for imagenet100, 491 for imagenette
                torch.save([x_advs, ts], f'x_advs_resnet_imagenette_{attack}_{num_batches_processed}.pt')
                x_advs = []
                ts = []
                gc.collect()
