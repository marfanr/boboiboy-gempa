from model.loader import ModelLoader
import argparse
from trainer import Trainer
from utils.DataLoader import DataLoader as InternalDataLoader
from torch.utils.data import DataLoader
from torchinfo import summary


def main():
    parser = argparse.ArgumentParser(description="Gempa")

    parser.add_argument("mode", help="train|test|ls", type=str)
    parser.add_argument("--hdf5", help="hdf5 file", type=str)
    parser.add_argument("--csv", help="csv file", type=str)
    parser.add_argument("--x_test", help="np file", type=str)
    parser.add_argument("--x_train", help="np file", type=str)
    parser.add_argument("--y_test", help="np file", type=str)
    parser.add_argument("--y_train", help="np file", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--batch", type=float, default=32)
    parser.add_argument("--max_epoch", type=float, default=15)
    parser.add_argument("--weight", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--writer", type=str)
    parser.add_argument("--count", type=int)
    parser.add_argument("--test_count", type=int)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--pos", type=int, default=0)
    parser.add_argument("--test_pos", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "ls":
        for i in ModelLoader.list_keys():
            print(i)
        return

    if args.model is None:
        raise ValueError("--model is required")

    model = ModelLoader.get(args.model)

    if args.mode == "info":
        m = model
        summary(
            m,
            input_size=(128, 3, 1000),
            col_names=["input_size", "output_size", "num_params"],
        )
        return

    #  create data loader from data
    data_loader = InternalDataLoader(
        args.hdf5,
        args.csv,
        args.x_test,
        args.x_train,
        args.y_test,
        args.y_train,
    )

    train_ds = data_loader.getDataset(1000, args.stride, args.count, args.pos, False)
    test_ds = data_loader.getDataset(
        1000, args.stride, args.test_count, args.test_pos, True
    )

    train_dataLoader = DataLoader(
        train_ds,
        args.batch,
        True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_dataLoader = DataLoader(
        test_ds,
        args.batch,
        False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    if args.mode == "train":
        trainer = Trainer(train_dataLoader, test_dataLoader, model)
        if args.hdf5 is not None:
            print("using hdf5")
        trainer.train(args.max_epoch, args.weight, args.out)


if __name__ == "__main__":
    main()
