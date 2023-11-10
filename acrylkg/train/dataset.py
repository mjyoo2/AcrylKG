import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings, label_encodings):
        """
        Initialize a custom dataset for text inputs and encodings.

        Parameters:
            encodings (dict): A dictionary containing the encoded inputs.
        """
        self.encodings = encodings
        self.label_encodings = label_encodings

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the encoded input and labels.
        """
        # Create an item dictionary with tensors for each encoding key
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        # Copy input_ids to labels for tasks like language modeling
        item["labels"] = self.label_encodings["input_ids"][idx]

        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.encodings["input_ids"])