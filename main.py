from model.loader import ModelLoader
import argparse
from trainer import Trainer
from utils.DataLoader import DataLoader as InternalDataLoader
from torch.utils.data import DataLoader


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
    parser.add_argument("--weight", type=str)
    parser.add_argument("--batch", type=float, default=32)
    parser.add_argument("--max_epoch", type=float, default=15)
    args = parser.parse_args()

    if args.mode == "ls":
        for i in ModelLoader.list_keys():
            print(i)
        return

    if args.model is None:
        raise ValueError("--model is required")

    #  create data loader from data
    data_loader = InternalDataLoader(
        args.hdf5,
        args.csv,
        args.x_test,
        args.x_train,
        args.y_test,
        args.y_train,
    )
    
    train_ds = data_loader.getDataset(1000, 100, 1000, 0, False)
    test_ds = data_loader.getDataset(1000, 100, 100, 0, True)
    
    train_dataLoader = DataLoader(train_ds, args.batch, True)
    test_dataLoader = DataLoader(test_ds, args.batch, False)
    model = ModelLoader.get(args.model)

    if args.mode == "train":
        trainer = Trainer(train_dataLoader, test_dataLoader, model)
        if args.hdf5 is not None:
            print("using hdf5")
        trainer.train(args.max_epoch)

if __name__ == "__main__":
    main()
