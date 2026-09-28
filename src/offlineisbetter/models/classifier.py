# offlineisbetter
# Copyright (c) 2026- offlineisbetter
#
# Classifier Model

import json

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Classifier head file
HEAD_PT = "classifier_head.pt"

# Classes JSON
CLASSES_JSON = "classes.json"


class ClassificationModel(nn.Module):
    """
    Text classification model.

    This model classifies input text into *one* of `N` mutually
        exclusive classes.  This is distinct from a labeling
        model, which classifies input text into *zero or more*
        of `N` nonexclusive classes.
    """

    def __init__(self, encoder, num_classes):
        """
        Initialize this classification model.
        """
        super().__init__()

        # Text encoder
        self.encoder = encoder

        # Construct output head
        self.head = nn.Linear(
            encoder.config.hidden_size,
            num_classes,
        ).bfloat16()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Compute the forward pass of this model, and return
            the loss if labels are specified.
        """
        input_ids, attention_mask = (
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
        )

        # Compute hidden state
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Compute logits
        logits = self.head(
            torch.mean(outputs.last_hidden_state, dim=1),
        )

        # If labels are specified, compute the loss
        loss = None
        if labels is not None:
            labels = torch.tensor(labels)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }

    def save(self, output_dir, classes):
        """
        Save this model.
        """
        # Merge and unload
        merged = self.encoder.merge_and_unload()

        # Save finetuned model
        merged.save_pretrained(
            output_dir,
            safe_serialization=True,
        )

        # Save classifier head
        torch.save(
            self.head.state_dict(),
            output_dir / HEAD_PT,
        )

        # Save classes
        with open(output_dir / CLASSES_JSON, "w") as f:
            json.dump(classes, f)

    @classmethod
    def load(cls, output_dir):
        """
        Load this model from its checkpoint.
        """
        # Load finetuned model
        encoder = AutoModelForMaskedLM.from_pretrained(
            output_dir,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Load classes
        with open(output_dir / CLASSES_JSON, "r") as f:
            classes = json.load(f)

        model = ClassificationModel(encoder, len(classes))

        # Load classifier head
        model.head.load_state_dict(torch.load(output_dir / HEAD_PT))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            output_dir,
            local_files_only=True,
            trust_remote_code=True,
        )

        return model, tokenizer
