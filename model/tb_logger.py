from typing import Any
from torch.utils.tensorboard import SummaryWriter


class CustomMetricsLogger():

    SAMPLE = 0
    EPOCH = 1
    BOTH = 2

    def __init__(self, tensorboard: SummaryWriter,  parent_tag: str):
        self.__writer = tensorboard
        self.__parent_tag = parent_tag
        self.__sample_pointer = 0
        self.__epoch_pointer = 0

    @property
    def sample(self) -> int:
        return self.__sample_pointer
    
    @property
    def epoch(self) -> int:
        return self.__epoch_pointer

    def sample_log(self, sub_tag: str, value: Any) -> None:
        tag = f"{self.__parent_tag}/{sub_tag}"
        self.__writer.add_scalar(tag, value, self.__sample_pointer)
    
    def epoch_log(self, sub_tag: str, value: Any) -> None:
        tag = f"{self.__parent_tag}/{sub_tag}"
        self.__writer.add_scalar(tag, value, self.__epoch_pointer)
    
    def count(self, mode: int, value: int = 1) -> None:
        if value < 0 or value > 2:
            raise RuntimeError("Value has to be: 0 (SAMPLE), 1 (EPOCH), 2 (BOTH).")
        
        if mode == CustomMetricsLogger.SAMPLE:
            self.__sample_pointer += value
            return
        
        if mode == CustomMetricsLogger.EPOCH:
            self.__epoch_pointer += value
            return
        
        if mode == CustomMetricsLogger.BOTH:
            self.__epoch_pointer += value
            self.__sample_pointer += value
            return

    def __enter__(self) -> "CustomMetricsLogger":
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        self.__writer.flush()