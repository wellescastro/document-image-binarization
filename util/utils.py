import shutil
import torch
import os


def save_checkpoint(state, is_best, checkpoint_dir, model_name):
    filename = os.path.join(checkpoint_dir, '{}-epoch-{}.pth'.format(model_name, state["epoch"]))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir + 'model_best.pth')

# TODO: built the checkpoint loading, here's a helper
def load_checkpoint(model, optimizer, load_path):
    """ loads state into model and optimizer and returns:
        epoch, best_precision, loss_train[]
    """
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        epoch = checkpoint['epoch']
        best_training_loss = checkpoint['best_training_loss']
        num_bad_epochs = checkpoint['num_bad_epochs']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(epoch, checkpoint['epoch']))
        return epoch, best_training_loss, num_bad_epochs
    else:
        print("=> no checkpoint found at '{}'".format(load_path))
        # epoch, best_precision, loss_train
        return 1, 0, []

def load_weights(model, load_path):
    """ loads state into model and optimizer and returns:
        epoch, best_precision, loss_train[]
    """
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(epoch, checkpoint['epoch']))
        return True
    else:
        print("=> no checkpoint found at '{}'".format(load_path))
        return False