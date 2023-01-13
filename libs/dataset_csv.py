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
}
