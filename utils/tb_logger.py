from typing import Any
from torch.utils.tensorboard import SummaryWriter


class CustomMetricsLogger():

    SAMPLE = 0
    EPOCH = 1
    BOTH = 2

    def __init__(self, tensorboard: SummaryWriter,  parent_tag: str):
        """This class can be used in a with-statement in order to enable tensorboard 
        logging.

        Args:
            tensorboard (SummaryWriter): The tensorboard writer.
            parent_tag (str): The root tag in tensorboard.
        """
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
        """Logs a given value under root_tag/sub_tag

        Args:
            sub_tag (str): The subtag of the value.
            value (Any): The value itself.
        """
        tag = f"{self.__parent_tag}/{sub_tag}"
        self.__writer.add_scalar(tag, value, self.__sample_pointer)
    
    def epoch_log(self, sub_tag: str, value: Any) -> None:
        """Logs a given value under root_tag/sub_tag

        Args:
            sub_tag (str): The subtag of the value.
            value (Any): The value itself.
        """
        tag = f"{self.__parent_tag}/{sub_tag}"
        self.__writer.add_scalar(tag, value, self.__epoch_pointer)
    
    def count(self, mode: int, value: int = 1) -> None:
        """This method moves the x-pointer of sample_log and/or epoch_log
        given the mode.

        Args:
            mode (int): Either SAMPLE, EPOCH or BOTH.
            value (int, optional): The amount of which the pointer is moved. 
            Defaults to 1.

        Raises:
            RuntimeError: Is thrown if a wrong mode is given.
        """
        if mode < 0 or mode > 2:
            raise RuntimeError(f"Mode has to be: 0 (SAMPLE), 1 (EPOCH), 2 (BOTH). Got {mode}")
        
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