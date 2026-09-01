
from transformers import logging

import torch
from transformers.trainer import _is_peft_model
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logger = logging.get_logger(__name__)   


class LazyBatchSamples:
    def __init__(self, epoch_iterator, num_batches):
        self.epoch_iterator = epoch_iterator
        self.num_batches = num_batches
        self.yielded = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.yielded >= self.num_batches:
            raise StopIteration
        batch = next(self.epoch_iterator)
        self.yielded += 1
        return batch

    def __len__(self):
        return self.num_batches


def get_batch_samples_lazy(self, epoch_iterator, num_batches, device):
    """Yield accumulation batches one at a time instead of preloading them on GPU."""
    return LazyBatchSamples(epoch_iterator, num_batches), None

def compute_loss_logging_labels(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    if self.model_accepts_loss_kwargs:
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
    label_num = torch.where(inputs['labels'] != -100,1,0).sum().item()
    input_len = inputs['input_ids'].shape[1]
    outputs = model(**inputs)
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            model_name = unwrapped_model.base_model.model._get_name()
        else:
            model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            loss = self.label_smoother(outputs, labels)
    else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    if (
        self.args.average_tokens_across_devices
        and (self.model_accepts_loss_kwargs or self.compute_loss_func)
        and num_items_in_batch is not None
    ):
        loss *= self.accelerator.num_processes

    return (loss, outputs) if return_outputs else loss
