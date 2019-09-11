"""Module defining models."""
from onmt.models.model_saver import build_model_saver, ModelSaver
from onmt.models.model import NMTModel
from onmt.models.uncond_model import UncondModel
from onmt.models.simple_fusion_model import SimpleFusionModel

__all__ = ["build_model_saver", "ModelSaver",
           "NMTModel", "SimpleFusionModel",
           "UncondModel",
           "check_sru_requirement"]
