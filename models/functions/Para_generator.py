# from .. import Alex_val_loss
import torch
import numpy as np


__all__ = ['para_generator']

def para_generator(device, model, data_loader, criterion, num_sample, args):

    b_size = len(model.avg_b_time)

    # criterion = Alex_val_loss(thresholds=[args.t1, args.t2], b_times=model.avg_b_time)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():

        entropy = np.zeros((b_size, num_sample))
        correct1 = np.zeros((b_size, num_sample))
        correct5 = np.zeros((b_size, num_sample))

        for i, (images, target) in enumerate(data_loader):
            images, target = images.to(device), target.to(device)

            # compute output
            output = model(images)
            b_entropy = criterion(output, target)[-b_size:]

            for j in range(b_size):
                entropy[j, i * args.batch_size: (i+1) * args.batch_size] = b_entropy[j].cpu().numpy()
                get_correct(output[j], target)
                correct1[j, i * args.batch_size: (i+1) * args.batch_size], correct5[j, i * args.batch_size: (i+1) * args.batch_size] = get_correct(output[j], target)

    return correct1, correct5, entropy

def get_correct(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_1 = correct[:1].float().sum(0, keepdim=True)
        correct_5 = correct[:5].float().sum(0, keepdim=True)

        return correct_1.cpu().numpy(), correct_5.cpu().numpy()