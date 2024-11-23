import torch

def compute_loss_preds(logits, labels):
    """ Computes the loss and accuracy for a batch of predictions. """
    loss = torch.nn.functional.cross_entropy(logits, labels)
    predictions = torch.argmax(logits, dim=1)
    correct_preds = (predictions == labels).sum().item()
    return loss, correct_preds

def evaluate_model(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        eval_loss, eval_correct_preds, eval_preds = 0.0, 0, 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            logits = model(embeddings)
            loss, correct_preds = compute_loss_preds(logits, labels)
            eval_loss += loss.item()
            eval_correct_preds += correct_preds
            eval_preds += labels.size(0)

    model.train()
    return eval_loss / len(dataloader), eval_correct_preds / eval_preds
