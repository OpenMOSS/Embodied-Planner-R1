# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import pickle
import numpy as np
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union, Any

import torch
import tensordict
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from verl.utils.py_functional import union_two_dict

__all__ = ['DataProto', 'union_tensor_dict']

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass

def pad_dataproto_to_divisor(data: 'DataProto', size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor.

    Args:
        size_divisor (int): Size divisor.

    Returns:
        data: (DataProto): The padded DataProto.
        pad_size (int): The size of padding applied.
    """
    assert isinstance(data, DataProto), 'data must be a DataProto'
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: 'DataProto', pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def fold_batch_dim(data: 'DataProto', new_batch_size):
    """
    Fold a batch dim from [bsz, xxx] into [new_bsz, bsz // new_bsz, xxx].
    """
    batch_size = len(data)

    assert batch_size % new_batch_size == 0

    tensor = data.batch
    non_tensor = data.non_tensor_batch

    folded_tensor = {key: val.view(new_batch_size, -1, *val.shape[1:]) for key, val in tensor.items()}
    folded_non_tensor = {key: np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:])) if isinstance(val, np.ndarray) else [val[i:i + batch_size // new_batch_size] for i in range(0, len(val), batch_size // new_batch_size)] for key, val in non_tensor.items()}

    return DataProto(batch=folded_tensor, non_tensor_batch=folded_non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: 'DataProto', batch_dims=2):
    """
    Unfold the first n dims as new batch dim.
    """
    tensor = data.batch
    non_tensor = data.non_tensor_batch

    unfolded_tensor = {key: val.view(-1, *val.shape[batch_dims:]) for key, val in tensor.items()}
    unfolded_non_tensor = {key: np.reshape(val, newshape=(-1, *val.shape[batch_dims:])) if isinstance(val, np.ndarray) else [item for sublist in val for item in sublist] for key, val in non_tensor.items()}

    return DataProto(batch=unfolded_tensor, non_tensor_batch=unfolded_non_tensor, meta_info=data.meta_info)


def collate_fn(x: list['DataProtoItem']):
    batch = {key: torch.stack([item.batch[key] for item in x]) for key in x[0].batch.keys()}
    non_tensor_batch = {key: [item.non_tensor_batch[key] for item in x] for key in x[0].non_tensor_batch.keys()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=x[0].meta_info)

@dataclass
class DataProtoItem:
    batch: Dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_batch: Dict[str, Any] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataProto:
    batch: Dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_batch: Dict[str, Any] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()

    def __len__(self):
        batch_len = next(iter(self.batch.values())).shape[0] if self.batch else 0
        non_tensor_len = len(next(iter(self.non_tensor_batch.values()))) if self.non_tensor_batch else 0

        if self.batch and self.non_tensor_batch:
            assert batch_len == non_tensor_len, "Lengths of batch and non_tensor_batch must be equal"

        if self.batch:
            return batch_len
        elif self.non_tensor_batch:
            return non_tensor_len
        else:
            return 0

    def __getitem__(self, item):
        tensor_data = {key: val[item] for key, val in self.batch.items()} if self.batch else {}
        non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()} if self.non_tensor_batch else {}
        return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def __getstate__(self):
        return self.batch, self.non_tensor_batch, self.meta_info

    def __setstate__(self, state):
        self.batch, self.non_tensor_batch, self.meta_info = state

    def save_to_disk(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> 'DataProto':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        size_of_tensors = sum(tensor.element_size() * tensor.numel() for tensor in self.batch.values()) / 1024**3
        size_of_non_tensors = sum(np.array(val).nbytes for val in self.non_tensor_batch.values()) / 1024**3 if self.non_tensor_batch else 0

        message = f'Size of tensors: {size_of_tensors} GB, size of non_tensors: {size_of_non_tensors} GB'

        if prefix:
            message = f'{prefix}, ' + message
        print(message)

    def check_consistency(self):
        batch_len = len(self.batch) if self.batch else 0
        non_tensor_len = len(next(iter(self.non_tensor_batch.values()))) if self.non_tensor_batch else 0
        if batch_len != 0 and non_tensor_len != 0:
            assert batch_len == non_tensor_len, "Lengths of batch and non_tensor_batch must be equal"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, Any]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            else:
                non_tensors[key] = val

        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors: Dict[str, Any] = None, meta_info: Dict[str, Any] = None):
        if non_tensors is None:
            non_tensors = {}
        if meta_info is None:
            meta_info = {}

        assert len(tensors) > 0 or len(non_tensors) > 0, 'At least one of tensors or non_tensors must not be empty'

        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> 'DataProto':
        """Move the batch to the specified device.

        Args:
            device (torch.device, str): Torch device.

        Returns:
            DataProto: The current DataProto with batch moved to the specified device.
        """
        if self.batch:
            self.batch = {key: val.to(device) for key, val in self.batch.items()}
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> 'DataProto':
        """Select a subset of the DataProto via batch_keys and meta_info_keys.

        Args:
            batch_keys (list, optional): A list of strings indicating the keys in batch to select.
            non_tensor_batch_keys (list, optional): A list of strings indicating the keys in non_tensor_batch to select.
            meta_info_keys (list, optional): A list of keys indicating the meta info to select.
            deepcopy (bool, optional): Whether to perform a deep copy of the selected data.

        Returns:
            DataProto: The DataProto with the selected batch_keys and meta_info_keys.
        """
        sub_batch = {key: val for key, val in self.batch.items() if batch_keys is None or key in batch_keys}
        sub_non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if non_tensor_batch_keys is None or key in non_tensor_batch_keys}
        sub_meta_info = {key: val for key, val in self.meta_info.items() if meta_info_keys is None or key in meta_info_keys}

        if deepcopy:
            sub_batch = copy.deepcopy(sub_batch)
            sub_non_tensor_batch = copy.deepcopy(sub_non_tensor_batch)
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=sub_non_tensor_batch, meta_info=sub_meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> 'DataProto':
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`.

        Args:
            batch_keys (list, optional): A list of strings indicating the keys in batch to pop.
            non_tensor_batch_keys (list, optional): A list of strings indicating the keys in non_tensor_batch to pop.
            meta_info_keys (list, optional): A list of keys indicating the meta info to pop.

        Returns:
            DataProto: The DataProto with the popped batch_keys and meta_info_keys.
        """
        if batch_keys is None:
            batch_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []

        tensors = {key: self.batch.pop(key) for key in batch_keys if key in self.batch}
        non_tensors = {key: self.non_tensor_batch.pop(key) for key in non_tensor_batch_keys if key in self.non_tensor_batch}
        meta_info = {key: self.meta_info.pop(key) for key in meta_info_keys if key in self.meta_info}

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> 'DataProto':
        """Rename the keys in the batch and non_tensor_batch.

        Args:
            old_keys (list or str): The old keys to be renamed.
            new_keys (list or str): The new keys to rename to.

        Returns:
            DataProto: The current DataProto with renamed keys.
        """

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f'keys must be a list or a string, but got {type(keys)}')
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(f'new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}')

        for old_key, new_key in zip(old_keys, new_keys):
            if old_key in self.batch:
                self.batch[new_key] = self.batch.pop(old_key)
            if old_key in self.non_tensor_batch:
                self.non_tensor_batch[new_key] = self.non_tensor_batch.pop(old_key)

        return self

    def union(self, other: 'DataProto') -> 'DataProto':
        """Union another DataProto with the current one.

        Args:
            other (DataProto): The other DataProto to be unioned.

        Returns:
            DataProto: The current DataProto after union.
        """
        for key, val in other.batch.items():
            if key in self.batch:
                assert torch.equal(self.batch[key], val), f'Conflict in batch key {key}'
            else:
                self.batch[key] = val

        for key, val in other.non_tensor_batch.items():
            if key in self.non_tensor_batch:
                assert np.array_equal(self.non_tensor_batch[key], val), f'Conflict in non_tensor_batch key {key}'
            else:
                self.non_tensor_batch[key] = val

        self.meta_info.update(other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto.

        Args:
            mini_batch_size (int): Mini-batch size when iterating the dataset.
            epochs (int): Number of epochs when iterating the dataset.
            seed (int, optional): Seed for reproducibility.
            dataloader_kwargs (dict, optional): Additional arguments for DataLoader.

        Returns:
            Iterator: An iterator that yields a mini-batch data at a time.
        """
        assert len(self) % mini_batch_size == 0, f"{len(self)} % {mini_batch_size} != 0"

        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        def collate_fn(batch):
            batch_data = {key: torch.stack([item.batch[key] for item in batch]) for key in batch[0].batch.keys()}
            non_tensor_data = {key: [item.non_tensor_batch[key] for item in batch] for key in batch[0].non_tensor_batch.keys()}
            return DataProto(batch=batch_data, non_tensor_batch=non_tensor_data, meta_info=batch[0].meta_info)

        train_dataloader = torch.utils.data.DataLoader(
            dataset=[DataProtoItem(batch=self.batch, non_tensor_batch=self.non_tensor_batch, meta_info=self.meta_info) for _ in range(len(self))],
            batch_size=mini_batch_size,
            collate_fn=collate_fn,
            generator=generator,
            **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List['DataProto']:
        """Split the batch among dim=0 into chunks.

        Args:
            chunks (int): The number of chunks to split on dim=0.

        Returns:
            List[DataProto]: A list of DataProto after splitting.
        """
        assert len(self) % chunks == 0, f'only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}.'

        batch_lst = [{} for _ in range(chunks)]
        non_tensor_batch_lst = [{} for _ in range(chunks)]

        for key, val in self.batch.items():
            split_tensors = torch.chunk(val, chunks, dim=0)
            for i, tensor in enumerate(split_tensors):
                batch_lst[i][key] = tensor

        for key, val in self.non_tensor_batch.items():
            if isinstance(val, np.ndarray):
                split_arrays = np.array_split(val, chunks)
                for i, array in enumerate(split_arrays):
                    non_tensor_batch_lst[i][key] = array
            else:
                split_lists = [val[i::chunks] for i in range(chunks)]
                for i, lst in enumerate(split_lists):
                    non_tensor_batch_lst[i][key] = lst

        return [DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info=self.meta_info) for batch, non_tensor in zip(batch_lst, non_tensor_batch_lst)]

    @staticmethod
    def concat(data: List['DataProto']) -> 'DataProto':
        """Concat a list of DataProto. The batch is concatenated among dim=0.

        Args:
            data (List[DataProto]): List of DataProto.

        Returns:
            DataProto: Concatenated DataProto.
        """
        if not data:
            return DataProto()

        batch = {}
        non_tensor_batch = {}
        meta_info = data[0].meta_info

        for d in data:
            for key, val in d.batch.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(val)
            for key, val in d.non_tensor_batch.items():
                if key not in non_tensor_batch:
                    non_tensor_batch[key] = []
                non_tensor_batch[key].append(val)

        for key in batch.keys():
            batch[key] = torch.cat(batch[key], dim=0)

        for key in non_tensor_batch.keys():
            if isinstance(non_tensor_batch[key][0], np.ndarray):
                non_tensor_batch[key] = np.concatenate(non_tensor_batch[key], axis=0)
            else:
                non_tensor_batch[key] = [item for sublist in non_tensor_batch[key] for item in sublist]

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def reorder(self, indices):
        """Reorder the batch and non_tensor_batch according to the given indices.

        Args:
            indices (torch.Tensor): The indices to reorder by.

        Note:
            This operation is in-place.
        """
        if self.batch:
            self.batch = {key: val[indices] for key, val in self.batch.items()}
        if self.non_tensor_batch:
            indices_np = indices.cpu().numpy()
            self.non_tensor_batch = {key: np.array(val)[indices_np].tolist() if isinstance(val, list) else val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        """Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch:
            if interleave:
                repeated_batch = {key: val.repeat_interleave(repeat_times, dim=0) for key, val in self.batch.items()}
            else:
                repeated_batch = {key: val.unsqueeze(0).expand(repeat_times, *val.shape).reshape(-1, *val.shape[1:]) for key, val in self.batch.items()}
        else:
            repeated_batch = {}

        if self.non_tensor_batch:
            if interleave:
                repeated_non_tensor_batch = {key: np.repeat(val, repeat_times, axis=0) if isinstance(val, np.ndarray) else val * repeat_times for key, val in self.non_tensor_batch.items()}
            else:
                repeated_non_tensor_batch = {key: np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1)) if isinstance(val, np.ndarray) else val * repeat_times for key, val in self.non_tensor_batch.items()}
        else:
            repeated_non_tensor_batch = {}

        return DataProto(batch=repeated_batch, non_tensor_batch=repeated_non_tensor_batch, meta_info=self.meta_info)


import ray


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture aims to eliminate actual data fetching on driver. By doing so, the driver doesn't have to wait
    for data so that asynchronous execution becomes possible. 
    DataProtoFuture contains a list of futures from another WorkerGroup of size world_size.
    - collect_fn is a Callable that reduces the list of futures to a DataProto
    - dispatch_fn is a Callable that partitions the DataProto into a list of DataProto of size world_size and then select

    Potential issue: we can optimize dispatch_fn(collect_fn) such that only needed data is fetched on destination
    - DataProtoFuture only supports directly passing from the output of a method to another input. You can't perform any
    operation on the DataProtoFuture in driver.
    """
    collect_fn: Callable
    futures: List[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: List[ray.ObjectRef]) -> 'DataProtoFuture':
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> List['DataProtoFuture']:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(collect_fn=self.collect_fn,
                                         dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks),
                                         futures=self.futures)
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto)
        output = self.collect_fn(output)  # select dp, concat
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        return output


from verl.utils.torch_functional import allgather_dict_tensors
import torch.distributed


def all_gather_data_proto(data: DataProto, process_group):
    # Note that this is an inplace operator just like torch.distributed.all_gather
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    prev_device = data.batch.device
    data.batch = data.batch.cuda(device=torch.cuda.current_device())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data.batch = data.batch.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
