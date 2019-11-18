import argparse
import signal

from common.dataset_management.dataset_server import DatasetServer


def main(args):
    server = DatasetServer(args)

    server.load_data()
    with server:
        print("Server Started")
        signal.pause()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that starts a server that supplies multiple programs with train data via shared memory")
    parser.add_argument("config", help="path to train config file that includes all paths that shall be loaded into memory")

    args = parser.parse_args()
    main(args)
