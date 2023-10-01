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
                test_dir: str = None,
                test_map_file: str = None,
                train_val_split: Tuple[int, int] = None,
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
        output = os.getcwd()
        if os.path.exists(os.path.join(output, 'data')):
            if os.listdir(os.path.join(output, 'data')):
                return
        else:
            os.makedirs(os.path.join(output, 'data'))
        gdown.download_folder(url, output= os.path.join(output, 'data'), quiet= False, use_cookies= False)
        with zipfile.ZipFile(os.path.join(output, 'data/training-images.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output, 'data'))
        with zipfile.ZipFile(os.path.join(output, 'data/dev-images.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output, 'data'))
        os.remove(os.path.join(output, 'data/training-images.zip'))
        os.remove(os.path.join(output, 'data/dev-images.zip'))

    def setup(self, stage: Optional[str] = None) -> None:

        if not self.setup_called:
            self.setup_called = True
            dataset = VQADataset(self.hparams.root_dir, map_file= self.hparams.map_file, data_dir= self.hparams.data_dir,
                                tokenizer= self.hparams.tokenizer, processor= self.hparams.processor, 
                                transforms= self.hparams.transforms, max_length= self.hparams.max_length
                                )
            if not self.hparams.train_val_split:
                self.data_train = dataset
                self.data_valid = VQADataset(self.hparams.root_dir, self.hparams.test_map_file, self.hparams.test_dir, 
                                             tokenizer= self.hparams.tokenizer, processor= self.hparams.processor, 
                                             transforms= self.hparams.transforms, max_length= self.hparams.max_length)
            # else:
            #     self.data_train, self.data_valid = random_split(
            #         dataset= dataset,
            #         lengths= self.hparams.train_val_split,
            #         generator= Generator().manual_seed(42)
            #     )
            if self.hparams.test_dir and not self.data_test:
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
            pin_memory= self.hparams.pin_memory,
            shuffle= False
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass