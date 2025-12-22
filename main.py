from model.loader import ModelLoader
import argparse
from trainer import Trainer
from utility.DataLoader import DataLoader as InternalDataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchinfo import summary
from utility.Writer import TensorWriter
from model.parser import ConfigParser
from utility.splitter import DataSplitter
import torch
import numpy as np

def collate_multi_station(batch):
    X_list, y_list = zip(*batch)
    x_wave, x_cords = zip(*X_list)
    # print(x_wave[0].shape[0])

    N_max = max(x.shape[0] for x in x_wave)
    # print(N_max)
    T, C = x_wave[0].shape[1:]
    B = len(x_wave)
    #
    X_wave_pad = torch.zeros(len(x_wave), N_max, *x_wave[0].shape[1:])
    X_cords_pad = torch.zeros(len(x_cords), N_max, *x_cords[0].shape[1:])
    station_mask = torch.zeros(B, N_max, dtype=torch.bool)
    #
    for i, x in enumerate(x_wave):
        n = x.shape[0]
        X_wave_pad[i, :n] = torch.from_numpy(x)
        station_mask[i, :n] = True
    #
    y = torch.from_numpy(np.stack(y_list, axis=0))
    #
    return X_wave_pad, X_cords_pad, station_mask, y


def main():
    parser = argparse.ArgumentParser(description="Gempa")

    subparser = parser.add_subparsers(
        dest="mode", required=True, help="train/test/ls/debug/split/info"
    )

    # data sorces

    # model
    parser.add_argument("--model", type=str)

    # training options
    train_parser = subparser.add_parser("train", help="train model")
    train_parser.add_argument("--model", type=str)
    train_parser.add_argument("--hdf5", help="hdf5 file", type=str)
    train_parser.add_argument("--csv", help="csv file", type=str)
    train_parser.add_argument("--x_test", help="np file", type=str)
    train_parser.add_argument("--x_train", help="np file", type=str)
    train_parser.add_argument("--y_test", help="np file", type=str)
    train_parser.add_argument("--y_train", help="np file", type=str)
    train_parser.add_argument("--meta_test", help="np file", type=str)
    train_parser.add_argument("--meta_train", help="np file", type=str)
    train_parser.add_argument("--train_npz", help="np file", type=str)
    train_parser.add_argument("--test_npz", help="np file", type=str)
    train_parser.add_argument("--batch", type=int, default=32)
    train_parser.add_argument("--max_epoch", type=float, default=15)
    train_parser.add_argument("--weight", type=str)
    train_parser.add_argument("--stride", type=int, default=500)
    train_parser.add_argument("--out", type=str)
    train_parser.add_argument("--writer", type=str)
    train_parser.add_argument("--count", type=int)
    train_parser.add_argument("--test_count", type=int)
    train_parser.add_argument("--pos", type=int, default=0)
    train_parser.add_argument("--test_pos", type=int, default=0)
    train_parser.add_argument("--dist", type=bool, default=False)
    train_parser.add_argument("--log", type=str)
    train_parser.add_argument("--noice", type=float, default=0.3)
    train_parser.add_argument("--normalize", type=bool, default=True)
    train_parser.add_argument("--compile", type=bool, default=False)
    train_parser.add_argument("--gpu_parallel", type=bool, default=False)
    train_parser.add_argument("--dataset", type=str)
    train_parser.add_argument("--multistation", type=bool, default=True)

    debug_parser = subparser.add_parser("debug", help="debug model")
    debug_parser.add_argument("--cfg", type=str)

    info_parser = subparser.add_parser("info", help="info model")
    info_parser.add_argument("--model", type=str)

    parser.add_argument("--compile", type=bool, default=True)
    train_parser.add_argument("--note", type=str, default="")

    # TODO: to be implemented

    args = parser.parse_args()

    if args.mode == "ls":
        for i in ModelLoader.list_keys():
            print(i)
        return

    if args.mode == "debug":
        cfg = args.cfg
        parser = ConfigParser(cfg)
        print(parser.parse())
        return

    if args.mode == "split":
        print("running splitter utility")
        splitter = DataSplitter(args.hdf5, args.csv, args.out)
        splitter.split()

    if args.model is None:
        raise ValueError("--model is required")

    model = ModelLoader.get(args.model)

    if args.mode == "info":
        summary(
            model,
        )
        return

    logger = None
    if args.log is not None:
        logger = TensorWriter(model.__class__.__name__, args.log, args.note)
        # sm = SystemMonitor(logger)
        # sm.start()

    #  create data loader from data
    data_loader = InternalDataLoader(args=args)

    train_ds = data_loader.getDataset(6000, args.stride, args.count, args.pos, False)
    test_ds = data_loader.getDataset(
        6000, args.stride, args.test_count, args.test_pos, True
    )

    # sample_weights = train_ds.get_sample_weights()
    # print(sample_weights)
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True,
    # )

    # train_ds.lab

    if args.hdf5 is not None and args.csv is not None:
        if args.gpu_parallel:
            train_dataLoader = DataLoader(
                train_ds,
                batch_size=args.batch,
                shuffle=True,
                num_workers=4,  # ← PENTING: parallel loading
                pin_memory=True,  # ← PENTING: untuk CUDA
                persistent_workers=True,  # ← workers tidak di-restart tiap epoch
                prefetch_factor=2,  # ← pre-load 2 batch ke depan
            )
            test_dataLoader = DataLoader(
                test_ds,
                batch_size=args.batch,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            train_dataLoader = DataLoader(
                train_ds, batch_size=args.batch,
                collate_fn=collate_multi_station
            )
            test_dataLoader = DataLoader(
                test_ds,
                batch_size=args.batch,
                shuffle=False,
                collate_fn=collate_multi_station
            )

    else:
        train_dataLoader = DataLoader(
            train_ds,
            args.batch,
            False,
            num_workers=0,
            pin_memory=True,
        )
        test_dataLoader = DataLoader(
            test_ds,
            args.batch,
            False,
            num_workers=0,
            pin_memory=True,
        )

    # Buat WeightedRandomSampler
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True,  # Dengan replacement untuk oversampling
    # )

    if args.mode == "train":
        trainer = Trainer(
            train_dataLoader,
            test_dataLoader,
            model,
            logger=logger,
            compile=args.compile,
        )
        if args.hdf5 is not None:
            print("using hdf5")

        trainer.train(
            max_epoch=args.max_epoch,
            weight=args.weight,
            output=args.out,
            distributed=args.gpu_parallel,
        )


if __name__ == "__main__":
    main()
