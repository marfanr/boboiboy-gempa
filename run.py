import argparse
from collector.stream import ObspyStreamWorker
from master import Master

def main():
    parser = argparse.ArgumentParser(
        description="Production DNN model"
    )

    mode_subparser = parser.add_subparsers(dest="mode", required=True)
    stream_parser = mode_subparser.add_parser("stream")
    stream_parser.add_argument("--station", required=True)
    stream_parser.add_argument("--id", required=True)

    master_parser = mode_subparser.add_parser("master")
    master_parser.add_argument("--worker", required=True, type=int)


    args = parser.parse_args()

    if args.mode == "stream":
        worker = ObspyStreamWorker(args.id, args.station)
        worker.run()

    elif args.mode == "master":
        master = Master(args.worker)
        master.run()




if __name__ == "__main__":
    main()