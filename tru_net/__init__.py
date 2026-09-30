"""TRU-Net: rede RNN multi-estado com canal limitado.
Arquitetura: estado -> transformacao -> estado (nao peso -> ativacao unica).
Sem Transformer. Tokenizacao por chunks. Backprop via BPTT.
"""
from .chunk_tokenizer import ChunkTokenizer
from .states import ViscousRNNModule, build_alpha_bands, diversity_loss
from .channel import LimitedChannel, symbol_loss
from .integrator import CommonSpace
from .readout import BrocaReadout
from .tru_net import TRUNet, TRUConfig, EpisodicMemory

__all__ = [
    "ChunkTokenizer",
    "ViscousRNNModule",
    "build_alpha_bands",
    "diversity_loss",
    "LimitedChannel",
    "symbol_loss",
    "CommonSpace",
    "BrocaReadout",
    "TRUNet",
    "TRUConfig",
    "EpisodicMemory",
]
