import numpy as np
from datasets import Dataset, Features, Value
from datasets import Image as DatasetImage
from PIL import Image

from children_drawings.data import ChildrenDrawingsDataModule


def _write_split(root, split_name: str, size: int = 4):
    image_dir = root / f"{split_name}_images"
    image_dir.mkdir(parents=True)
    image_paths = []
    categories = ["집", "나무", "남자사람", "여자사람"]
    genders = ["남", "여", "남", "여"]

    for index in range(size):
        image_path = image_dir / f"{index}.png"
        pixels = np.full((32, 32, 3), fill_value=index * 20, dtype=np.uint8)
        Image.fromarray(pixels).save(image_path)
        image_paths.append(str(image_path))

    dataset = Dataset.from_dict(
        {
            "image": image_paths,
            "category": categories[:size],
            "age": [7 + index for index in range(size)],
            "gender": genders[:size],
            "id": [str(index) for index in range(size)],
        },
        features=Features(
            {
                "image": DatasetImage(),
                "category": Value("string"),
                "age": Value("int64"),
                "gender": Value("string"),
                "id": Value("string"),
            }
        ),
    )
    dataset.save_to_disk(str(root / split_name))


def test_datamodule_returns_one_batch(tmp_path):
    _write_split(tmp_path, "train")
    _write_split(tmp_path, "validation")

    datamodule = ChildrenDrawingsDataModule(
        data_root=str(tmp_path),
        batch_size=2,
        num_workers=0,
    )
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))

    assert batch["image"].shape == (2, 3, 300, 300)
    assert batch["category"].shape == (2,)
    assert batch["age"].shape == (2,)
    assert batch["gender"].shape == (2,)
