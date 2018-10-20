import shutil
import torch
import os


def save_checkpoint(state, is_best, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, 'auto_encoder-epoch-{}.pth'.format(state["epoch"]))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir + 'model_best.pth')

# TODO: built the checkpoint loading, here's a helper
def load_checkpoint(checkpoint, model, optimizer, load_path):
    """ loads state into model and optimizer and returns:
        epoch, best_precision, loss_train[]
    """
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        loss_train = checkpoint['loss_train']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(epoch, checkpoint['epoch']))
        return epoch, best_prec, loss_train
    else:
        print("=> no checkpoint found at '{}'".format(load_path))
        # epoch, best_precision, loss_train
        return 1, 0, []