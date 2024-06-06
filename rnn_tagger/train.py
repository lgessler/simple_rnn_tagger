import shutil
import time

import torch


# run one epoch of training
def train(model, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    words_correct = 0
    words = 0
    step = 0
    # Iterate over data.
    for batch in train_loader:
        # zero grad
        model.zero_grad()

        output = model(**batch)
        loss = output["loss"]

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        words_correct += output["num_words_correct"]
        words += output["num_words"]
        step += 1
        if step % 10 == 0:
            acc = (words_correct / words) * 100
            print('Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss, acc, words_correct, words))
    loss = running_loss / words
    acc = (words_correct / words) * 100
    print('Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss, acc, words_correct, words))
    return loss, acc


def validate(model, val_loader):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    words_correct = 0
    words = 0
    # Iterate over data.
    for batch in val_loader:
        # zero grad
        model.zero_grad()

        output = model(**batch)

        # statistics
        running_loss += output["loss"]
        words_correct += output["num_words_correct"]
        words += output["num_words"]
    loss = running_loss / words
    acc = (words_correct / words) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss, acc, words_correct, words))
    return loss, acc


def train_model(model, data_loaders, optimizer, device, num_epochs=25):
    model = model.to(device)

    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_begin = time.time()
        train_loss, train_acc = train(model, data_loaders['train'], optimizer)
        train_time = time.time() - train_begin
        print('Epoch Train Time: {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
        print('Train Loss', train_loss, epoch)
        print('Train Accuracy', train_acc, epoch)

        validation_begin = time.time()
        val_loss, val_acc = validate(model, data_loaders['dev'])
        validation_time = time.time() - validation_begin
        print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
        print('Validation Loss', val_loss, epoch)
        print('Validation Accuracy', val_acc, epoch)

        # deep copy the model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        save_checkpoint("models", {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
        }, is_best)

    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def save_checkpoint(save_dir, state, is_best):
    savepath = save_dir + '/' + 'checkpoint.pth.tar'
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, save_dir + '/' + 'model_best.pth.tar')
