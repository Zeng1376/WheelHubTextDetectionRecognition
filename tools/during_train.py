import torch


def train_one_epoch(epoch_index, training_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.
    loss_fn.cuda()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    model.train()

    correct = 0

    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs.to(device='cuda')
        labels.to(device='cuda')

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        outputs.cuda()

        # Compute the loss and its gradients
        # print(outputs.shape, labels)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    last_loss = running_loss / len(training_loader)  # loss per batch
    print(' batch {} loss: {}'.format(len(training_loader), last_loss))
    print(f'precision:{correct / len(training_loader.dataset)}')
    # tb_x = epoch_index * len(training_loader) + i + 1
    # running_loss = 0.

    print("-----during train------")
    print(outputs[0, :])
    print(torch.argmax(outputs, dim=1))
    print(labels)
    # TODO write a txt logger

    return last_loss


def test_one_epoch(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    running_loss = 0.
    correct = 0

    model.eval()

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            vdata, label = data
            pred = model(vdata)
            running_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    running_loss /= len(data_loader)
    correct /= size
    print(f"During Test: \n Accuracy:{correct}, Avg loss:{running_loss}")

    return running_loss, correct
