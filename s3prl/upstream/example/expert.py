from collections import OrderedDict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from s3prl.upstream.example.conformer import conformer_base_architecture, S2TConformerEncoder
import s3prl.hub as hub

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "conformer"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )
        
        x = torch.load(ckpt)

        # pass arg file 
        y = x['args']
        conformer_base_architecture(y) # updates y with conformer_base_architecture args
        y.conv_version = 's2t_transformer' # this is required for the conformer encoder to work

        # rename keys by removing 'encoder.' prefix
        x['model'] = OrderedDict((k.replace('encoder.', ''), v) for k, v in x['model'].items())

        self.fbank_featurizer = getattr(hub, 'fbank')()  
        self.backbone = S2TConformerEncoder(y)
        self.backbone.load_state_dict(x['model'], strict=False)

        '''
        # The model needs to be a nn.Module for finetuning, not required for representation extraction
        self.model1 = nn.Linear(1, HIDDEN_DIM)
        self.model2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        '''

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 64 # downsampling will take place in the conformer_base_architecture

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        # src_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(wavs[0].device)
        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        # wavs: (batch_size, max_len, 1)

        # write code to get lfbe features using torch kaldis lfbematrix

        wavs = wavs.transpose(1, 2)
        wavs = self.fbank_featurizer(wavs)['last_hidden_state'][:,:1000,:80] 
        # https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/README.md only first 80 mel bins are used and only first 1000 frames are used
        src_lengths = wavs.shape[1]*torch.ones(wavs.shape[0]).long().to(wavs.device)
        out  = self.backbone(wavs, src_lengths, return_all_hiddens=True)

        return {
            "hidden_states": out["encoder_states"],
            "last_hidden_state": out["encoder_out"],
        }

        '''

        hidden = self.model1(wavs)
        # hidden: (batch_size, max_len, hidden_dim)

        feature = self.model2(hidden)
        # feature: (batch_size, max_len, hidden_dim)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": [hidden, feature],
            "PR": [hidden, feature],
            "ASR": [hidden, feature],
            "QbE": [hidden, feature],
            "SID": [hidden, feature],
            "ASV": [hidden, feature],
            "SD": [hidden, feature],
            "ER": [hidden, feature],
            "SF": [hidden, feature],
            "SE": [hidden, feature],
            "SS": [hidden, feature],
            "secret": [hidden, feature],
        }
        '''
