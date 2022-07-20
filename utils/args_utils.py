import argparse


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--data-dir", type=str, default="D:/won/data")
    parser.add_argument("--img-size", nargs="+", type=int, default=[416, 416])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--name", type=str, default="voc/2007")
    parser.add_argument("--lambda-yx", type=float, default=1e-1)
    parser.add_argument("--lambda-hw", type=float, default=1e-5)
    parser.add_argument("--lambda-obj", type=float, default=1e-1)
    parser.add_argument("--lambda-nobj", type=float, default=1e-4)
    parser.add_argument("--lambda-cls", type=float, default=1e-3)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    return args
