import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches=None
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader)

    for batch_idx, (images, labels) in enumerate(progress_bar):

        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        current_loss = running_loss / (batch_idx + 1)
        current_acc = correct / total

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.4f}"
        )

    return running_loss / (batch_idx + 1), correct / total

def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = correct / total

    return loss, accuracy