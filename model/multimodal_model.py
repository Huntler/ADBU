from model.base_model import BaseModel


class MultimodalModel(BaseModel):
    def __init__(self, tag: str, log: bool = True) -> None:
        super(MultimodalModel).__init__(tag, log)

        # TODO: add image model
        # TODO: add sensor model
        # TODO: add concatenate
        # TODO: add LSTM
        # TODO: add classifier output

        # TODO: define optimizer, loss function and scheduler as BaseModel needs