import gc
import json
import math
import os
import random
import time
from collections import defaultdict

gc.enable()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


class Config:
    # general
    data_path = "/content/drive/MyDrive/kaggle/commonlit/input/"
    output_path = "/content/drive/MyDrive/kaggle/commonlit/output/"
    seed = 42
    data_split_seed = 2021

    # optimizer
    optimizer_name = "AdamW"
    optimizer_default_kwargs = {
        "lr": 2e-5,
        "weight_decay": 0.1,
        "betas": (0.9, 0.98),
        "eps": 1e-06,
    }
    lr_decay_factor = 0.975
    lr_layer_base = 2e-5
    lr_no_pretrained = 1e-3
    weight_decay = 0.01

    # scheduler
    scheduler_name = "cosine_warmup"
    warmup_proportion = 0.1

    # training
    base_model_name = "roberta-base"
    epochs = 3
    evaluate_interval = 200
    train_bs = 8
    valid_bs = 16
    test_bs = 8
    multisample_dropout = True
    multisample_times = 5
    dropout_rate = 0.1
    output_hidden_states = True
    num_labels = 1
    max_len = 256


def random_seed(seed):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class RoBERTaDataSet(Dataset):
    def __init__(self, sentences, tokenizer, max_len, targets=None):

        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.targets = targets

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        tok = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = torch.tensor(tok["input_ids"], dtype=torch.long)
        mask = torch.tensor(tok["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(tok["token_type_ids"], dtype=torch.long)

        if self.targets is None:
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
            }
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.double)

            return {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": token_type_ids,
                "label": target,
            }


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class DataProcessor:
    def __init__(self, config, raw_data: pd.DataFrame, mode, target_col_name="target"):
        self.config = config
        self.raw_data = raw_data
        self.mode = mode
        self.target_col_name = target_col_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

    def loader(self):
        if self.mode == "train":
            dataset = RoBERTaDataSet(
                self.raw_data["excerpt"].values,
                self.tokenizer,
                self.config.max_len,
                self.raw_data[self.target_col_name].values,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.config.train_bs,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=4,
            )
        elif self.mode == "valid":
            dataset = RoBERTaDataSet(
                self.raw_data["excerpt"].values,
                self.tokenizer,
                self.config.max_len,
                self.raw_data[self.target_col_name].values,
            )
            loader = DataLoader(
                dataset, batch_size=self.config.valid_bs, pin_memory=True, num_workers=4
            )
        elif self.mode == "test":
            dataset = RoBERTaDataSet(
                self.raw_data["excerpt"].values, self.tokenizer, self.config.max_len
            )
            loader = DataLoader(
                dataset, batch_size=self.config.test_bs, pin_memory=True, num_workers=4
            )
        else:
            raise ValueError("Not a valid mode for DataProcessor.")

        return loader


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(self.config.base_model_name)
        self.model_config.update({"num_labels": self.config.num_labels})
        self.huggingface_model = AutoModel.from_pretrained(
            self.config.base_model_name,
            output_hidden_states=self.config.output_hidden_states,
        )
        self.layer_norm = nn.LayerNorm(self.model_config.hidden_size)
        # if self.config.multisample_dropout:
        #     self.dropouts = nn.ModuleList([
        #         nn.Dropout(self.config.dropout_rate) for _ in range(self.config.multisample_times)
        #     ])
        # else:
        #     self.dropouts = nn.ModuleList([nn.Dropout(self.config.dropout_rate)])
        self.regressor1 = nn.Linear(self.model_config.hidden_size * 3, 256)
        self.regressor2 = nn.Linear(256, 1)

        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor1)
        self._init_weights(self.regressor2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.model_config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            module.data.normal_(mean=0.0, std=self.model_config.initializer_range)

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None
    ):
        outputs = self.huggingface_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        last_hidden_state = outputs["hidden_states"][-1]  # (8, 256, 1024)
        last_hidden_state = self.layer_norm(last_hidden_state)

        last2_hidden_state = outputs["hidden_states"][-2]  # (8, 256, 1024)
        last2_hidden_state = self.layer_norm(last2_hidden_state)

        last3_hidden_state = outputs["hidden_states"][-3]  # (8, 256, 1024)
        last3_hidden_state = self.layer_norm(last3_hidden_state)

        last_hidden_state_mean = torch.mean(last_hidden_state, 1)
        last2_hidden_state_mean = torch.mean(last2_hidden_state, 1)
        last3_hidden_state_mean = torch.mean(last3_hidden_state, 1)
        mean_pool_concat = torch.cat(
            (last_hidden_state_mean, last2_hidden_state_mean, last3_hidden_state_mean),
            1,
        )

        # multi-sample dropout
        # for i, dropout in enumerate(self.dropouts):
        #     if i == 0:
        #         logits = self.regressor2(self.regressor1(dropout(mean_pool_concat)))
        #     else:
        #         logits += self.regressor2(self.regressor1(dropout(mean_pool_concat)))

        # logits /= len(self.dropouts)
        logits = self.regressor2(self.regressor1(mean_pool_concat))
        return logits


class Trainer:
    def __init__(self, model, config, fold):
        self.model = model
        self.config = config
        self.loss_fn = torch.nn.MSELoss()
        self.history = {"val_loss": [], "best_val_loss": np.inf}
        self.state = {"fold": fold, "epoch": 0}

    def get_optimizer_params(self):
        # differential learning rate and weight decay
        param_optimizer = list(self.model.named_parameters())
        learning_rate = self.config.lr_layer_base
        no_decay = ["bias", "gamma", "beta"]  # weight_decay対象外のパラメータたち

        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "huggingface_model" not in n
                ],
                "lr": self.config.lr_no_pretrained,
            },
            {
                "params": [
                    p
                    for n, p in self.model.huggingface_model.named_parameters()
                    if any(nd in n for nd in no_decay) and "layer" not in n
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.model.huggingface_model.named_parameters()
                    if not any(nd in n for nd in no_decay) and "layer" not in n
                ],
                "weight_decay": self.config.weight_decay,
            },
        ]

        decay_factor = self.config.lr_decay_factor
        num_layers = self.model.model_config.num_hidden_layers
        for i in range(1, num_layers + 1):
            optimizer_parameters.extend(
                [
                    # layer.{}. としないとlayer20などがlayer2で引っかかってしまう
                    {
                        "params": [
                            p
                            for n, p in self.model.huggingface_model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                            and f"layer.{num_layers - i}." in n
                        ],
                        "weight_decay": self.config.weight_decay,
                        "lr": learning_rate * (decay_factor ** i),
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.huggingface_model.named_parameters()
                            if any(nd in n for nd in no_decay)
                            and f"layer.{num_layers - i}." in n
                        ],
                        "weight_decay": 0.0,
                        "lr": learning_rate * (decay_factor ** i),
                    },
                ]
            )
        return optimizer_parameters

    def get_optimizer(self, optimizer_name, optimizer_default_kwargs):
        optimizer_grouped_parameters = self.get_optimizer_params()

        if optimizer_name == "LAMB":
            optimizer = Lamb(optimizer_grouped_parameters, **optimizer_default_kwargs)
            return optimizer
        elif optimizer_name == "Adam":
            from torch.optim import Adam

            optimizer = Adam(optimizer_grouped_parameters, **optimizer_default_kwargs)
            return optimizer
        elif optimizer_name == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, **optimizer_default_kwargs)
            return optimizer
        else:
            raise Exception("Unknown optimizer: {}".format(optimizer_name))

    def get_scheduler(self, optimizer, scheduler_name, scheduler_kwargs):
        if scheduler_name == "step":
            scheduler = lr_scheduler.MultiStepLR(optimizer, **scheduler_kwargs)
        elif scheduler_name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        elif scheduler_name == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_kwargs)
        elif scheduler_name == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(optimizer, **scheduler_kwargs)
        else:
            raise Exception("Unknown lr scheduler: {}".format(scheduler_name))
        return scheduler

    def train_one_epoch(self, train_loader, valid_loader, optimizer, scheduler):
        loss_meter = AverageMeter()
        count = 0
        self.model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids, attention_mask, token_type_ids, labels = (
                batch_data["input_ids"].cuda(),
                batch_data["attention_mask"].cuda(),
                batch_data["token_type_ids"].cuda(),
                batch_data["label"].cuda(),
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            count += labels.size(0)

            outputs = outputs.view(-1).to(labels.dtype)
            loss = torch.sqrt(self.loss_fn(outputs, labels.view(-1)))
            loss_meter.update(loss.item(), input_ids.size(0))

            loss.backward()
            optimizer.step()
            scheduler.step()

            # 理由不明だが...
            if batch_idx % self.config.evaluate_interval == 0:
                self.model.eval()

        # compute train loss
        train_len = len(train_loader.dataset)
        _s = str(len(str(train_len)))  # train_lenの桁数
        msg = [
            ("Epoch: [{}] [{: >" + _s + "}/{} ({: >3.0f}%)]").format(
                self.state["epoch"], count, train_len, 100 * count / train_len
            ),
            "train_loss: {: >4.5f}".format(loss_meter.avg),
        ]
        msg = "   ".join(msg)

        # compute val loss
        msg += self.evaluate(valid_loader)

        if self.history["val_loss"][-1] < self.history["best_val_loss"]:
            msg += "   <-----------------------   best valid loss was updated!!"
            self.history["best_val_loss"] = self.history["val_loss"][-1]
            torch.save(
                self.model.state_dict(),
                self.config.output_path + f"model{self.state['fold']}.bin",
            )

        print(msg)

    # train_one_epoch内でevaluate_intervalごとに呼ばれる
    # ModelAPIでpathを読み込み予測し、metricを計算
    def evaluate(self, valid_loader):
        loss_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(valid_loader):
                input_ids, attention_mask, token_type_ids, labels = (
                    batch_data["input_ids"].cuda(),
                    batch_data["attention_mask"].cuda(),
                    batch_data["token_type_ids"].cuda(),
                    batch_data["label"].cuda(),
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                outputs = outputs.view(-1).to(labels.dtype)
                loss = torch.sqrt(self.loss_fn(outputs, labels.view(-1)))
                loss_meter.update(loss.item(), input_ids.size(0))

        msg = "   valid_loss: {: >4.5f}".format(loss_meter.avg)
        self.history["val_loss"].append(loss_meter.avg)
        return msg

    def train_loop(self, train, valid):
        self.model = self.model.cuda()
        train_processor = DataProcessor(self.config, train, mode="train")
        valid_processor = DataProcessor(self.config, valid, mode="valid")

        num_update_steps_per_epoch = len(train_processor.loader())
        max_train_steps = self.config.epochs * num_update_steps_per_epoch
        warmup_steps = math.ceil(max_train_steps * self.config.warmup_proportion)
        scheduler_kwargs = {
            "num_warmup_steps": warmup_steps,
            "num_training_steps": max_train_steps,
        }

        optimizer = self.get_optimizer(
            self.config.optimizer_name, self.config.optimizer_default_kwargs
        )
        scheduler = self.get_scheduler(
            optimizer, self.config.scheduler_name, scheduler_kwargs
        )

        train_loader = train_processor.loader()
        valid_loader = valid_processor.loader()

        for epoch in range(self.config.epochs):
            # train one epoch
            self.train_one_epoch(train_loader, valid_loader, optimizer, scheduler)
            self.state["epoch"] += 1


class ModelAPI:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        self.model = Model(self.config)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, df):
        data_processor = DataProcessor(
            self.config, df, mode="test"
        )  # validのこともあるが、ラベルを使わないのでtestでよい
        self.model.cuda()
        self.model.eval()
        outputs_list = []
        with torch.no_grad():
            for batch_data in data_processor.loader():
                input_ids, attention_mask, token_type_ids = (
                    batch_data["input_ids"].cuda(),
                    batch_data["attention_mask"].cuda(),
                    batch_data["token_type_ids"].cuda(),
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                outputs = outputs.view(-1).cpu().detach().numpy()
                outputs_list.append(outputs)
        return np.concatenate(outputs_list)


class Runner:
    def __init__(self, config):
        self.config = config

    def _create_folds(self, data: pd.DataFrame, num_splits):
        data["kfold"] = -1
        kf = KFold(
            n_splits=num_splits, shuffle=True, random_state=self.config.data_split_seed
        )
        for f, (t_, v_) in enumerate(kf.split(X=data)):
            data.loc[v_, "kfold"] = f
        return data

    def run_fold(self, fold, train_df, valid_df):
        model = Model(self.config)
        trainer = Trainer(model, self.config, fold)
        trainer.train_loop(train_df, valid_df)
        valid_pred = ModelAPI(
            self.config, self.config.output_path + f"model{fold}.bin"
        ).predict(valid_df)
        return valid_pred, valid_df["target"].values

    def run_5folds_cv(self):
        train = pd.read_csv(self.config.data_path + "train.csv")
        train["is_modern"] = train["license"].isnull().map({True: 0, False: 1})
        test = pd.read_csv(self.config.data_path + "test.csv")
        train = self._create_folds(train, 5)

        valid_preds, valid_trues = [], []
        valid_moderns = []
        for fold in range(5):
            print("-" * 100)
            print(f"FOLD: {fold}")
            train_cv, valid_cv = (
                train[train["kfold"] != fold],
                train[train["kfold"] == fold],
            )
            valid_modern = valid_cv["is_modern"]
            valid_pred, valid_true = self.run_fold(fold, train_cv, valid_cv)
            valid_preds.append(valid_pred)
            valid_trues.append(valid_true)
            valid_moderns.append(valid_modern)

        valid_moderns = np.concatenate(valid_moderns)
        valid_preds = np.concatenate(valid_preds)
        valid_trues = np.concatenate(valid_trues)
        rmse = self.rmse(valid_preds, valid_trues)
        print(f"5fold-CV RMSE: {rmse}")
        return valid_moderns, valid_preds, valid_trues

    @staticmethod
    def rmse(pred, true):
        return np.sqrt(((pred - true) ** 2).mean())
