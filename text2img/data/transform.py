import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from tqdm import tqdm


def text_tokens_to_representation(tokens_tensor, language_model, batch_size=32):
    """Computes the representation (last hidden state of the language model) of token tensors by batch"""
    tokens_dataset = TensorDataset(tokens_tensor)
    tokens_data_loader = DataLoader(tokens_dataset, batch_size=batch_size, drop_last=False)

    batched_text_sentence_representations = []

    with torch.no_grad():
        for batch in tqdm(tokens_data_loader):
            lm_output = language_model(batch[0])
            # Use the last hidden state of the language model as the representation of the sentence
            # Representation of shape (batch_size, seq_len, hidden_dim)
            batched_text_sentence_representations.append(lm_output[0])

    text_sentence_representations = torch.cat(batched_text_sentence_representations, dim=0)

    return text_sentence_representations


def image_label_to_representation(labels_tensor, embedding_module, batch_size=32):
    """Compute the representations / embeddings of the image class labels by batch"""
    one_hot_labels_tensor = F.one_hot(labels_tensor).type(torch.FloatTensor)

    one_hot_labels_tensor_dataset = TensorDataset(one_hot_labels_tensor)
    one_hot_labels_tensor_data_loader = DataLoader(one_hot_labels_tensor_dataset,
                                                   batch_size=batch_size, drop_last=False)
    batched_labels_embeddings = []

    with torch.no_grad():
        for batch in tqdm(one_hot_labels_tensor_data_loader):
            batched_labels_embeddings.append(batch[0] @ embedding_module.weight.t())

    labels_embeddings = torch.cat(batched_labels_embeddings, dim=0)

    return labels_embeddings
