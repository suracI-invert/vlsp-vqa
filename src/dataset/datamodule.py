from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Generator

from src.dataset.components.datasets import VQADataset
import gdown
import os
import json

import zipfile

class VQADataModule(LightningDataModule):
    def __init__(self, root_dir: str,
                data_dir: str,
                map_file: str,
                val_dir: str = None,
                val_map_file: str = None,
                test_dir: str = None,
                test_map_file: str = None,
                tokenizer = None,
                processor= None,
                max_length: int = 512,
                batch_size: int = 1,
                num_workers: int = 0,
                pin_memory: bool = False,
                transforms= None,
                collate_fn= None,
                sampler= None,
            ) -> None:
        super().__init__()

        self.save_hyperparameters(logger= False)
        self.prepare_data_per_node = False

        self.data_train: Optional[Dataset] = None
        self.data_valid: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


        self.setup_called = False

    @property
    def num_classes(self) -> int:
        pass

    def prepare_data(self) -> None:
        url = 'https://drive.google.com/drive/folders/1zNzpR8XCVwweRGExnjotwrJuCnhxfiTO'
        if os.path.exists(self.hparams.root_dir):
            if os.listdir(self.hparams.root_dir):
                return
        else:
            os.makedirs(self.hparams.root_dir)
        gdown.download_folder(url, output= self.hparams.root_dir, quiet= False, use_cookies= False)
        with zipfile.ZipFile(os.path.join(self.hparams.root_dir, 'training-images.zip'), 'r') as zip_ref:
            zip_ref.extractall(self.hparams.root_dir)
        with zipfile.ZipFile(os.path.join(self.hparams.root_dir, 'dev-images.zip'), 'r') as zip_ref:
            zip_ref.extractall(self.hparams.root_dir)
        with zipfile.ZipFile(os.path.join(self.hparams.root_dir, 'test-images.zip'), 'r') as zip_ref:
            zip_ref.extractall(self.hparams.root_dir)
        os.remove(os.path.join(self.hparams.root_dir, 'training-images.zip'))
        os.remove(os.path.join(self.hparams.root_dir, 'dev-images.zip'))
        os.remove(os.path.join(self.hparams.root_dir, 'test-images.zip'))

    def setup(self, stage: Optional[str] = None) -> None:

        if not self.setup_called:
            self.setup_called = True
            self.data_train = VQADataset(self.hparams.root_dir, map_file= self.hparams.map_file, data_dir= self.hparams.data_dir,
                                tokenizer= self.hparams.tokenizer, processor= self.hparams.processor, 
                                transforms= self.hparams.transforms, max_length= self.hparams.max_length
                                )
            self.data_valid = VQADataset(self.hparams.root_dir, self.hparams.val_map_file, self.hparams.val_dir, 
                                             tokenizer= self.hparams.tokenizer, processor= self.hparams.processor, 
                                             transforms= self.hparams.transforms, max_length= self.hparams.max_length)
            self.data_test = VQADataset(self.hparams.root_dir, self.hparams.test_map_file, self.hparams.test_dir, 
                                            tokenizer= self.hparams.tokenizer, processor= self.hparams.processor, 
                                            transforms= self.hparams.transforms, max_length= self.hparams.max_length)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset= self.data_train,
            batch_size= self.hparams.batch_size,
            num_workers= self.hparams.num_workers,
            collate_fn= self.hparams.collate_fn,
            sampler= self.hparams.sampler,
            pin_memory= self.hparams.pin_memory,
            shuffle= True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset= self.data_valid,
            batch_size= self.hparams.batch_size,
            num_workers= self.hparams.num_workers,
            collate_fn= self.hparams.collate_fn,
            sampler= self.hparams.sampler,
            pin_memory= self.hparams.pin_memory,
            shuffle= False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset= self.data_test,
            batch_size= self.hparams.batch_size,
            num_workers= self.hparams.num_workers,
            collate_fn= self.hparams.collate_fn,
            pin_memory= self.hparams.pin_memory,
            shuffle= False
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass