import dataclasses
from logging import getLogger

logger = getLogger(__name__)

__all__ = ["DATASET_CSVS"]


@dataclasses.dataclass(frozen=True)
class DatasetCSV:
    train: str
    test: str


DATASET_CSVS = {
    # paths from `src` directory
    "Jung": DatasetCSV(
        train="./csv/Jung/train.csv",
        test="./csv/Jung/test.csv",
    ),
    "Kligler": DatasetCSV(
        train="./csv/Kligler/train.csv",
        test="./csv/Kligler/test.csv",
    ),
    "Shadoc": DatasetCSV(
        train="./csv/Shadoc/train.csv",
        test="./csv/Shadoc/test.csv",
    ),
    "Adobe": DatasetCSV(
        train="./csv/Adobe/train.csv",
        test="./csv/Adobe/test.csv",
    ),
    "HS": DatasetCSV(
        train="./csv/HS/train.csv",
        test="./csv/HS/test.csv",
    ),
    "Shadoc": DatasetCSV(
        train="./csv/Shadoc/train.csv",
        test="./csv/Shadoc/test.csv",
    ),
}
