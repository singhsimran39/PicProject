import argparse
import tensorflow as tf
from model_oo import PicProject

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=3)
parser.add_argument('-input_height', type=int, default=200)
parser.add_argument('-input_width', type=int, default=200)
parser.add_argument('-input_channels', type=int, default=3)
parser.add_argument('-fc1_dim', type=int, default=1000)
parser.add_argument('-fc2_dim', type=int, default=2)
parser.add_argument('-out_dim', type=int, default=2)
parser.add_argument('-num_maps', type=int, default=16)
parser.add_argument('-epochs', type=int, default=1)
parser.add_argument('-learning_rate', type=float, default=0.0001)
parser.add_argument('-train_path')
parser.add_argument('-test_path')
parser.add_argument('-model_name', default='picProject.model')
args = parser.parse_args()

def main():
    with tf.Session() as sess:

        pp = PicProject(sess,
                        batch_size=args.batch_size,
                        input_height=args.input_height,
                        input_width=args.input_width,
                        input_channels=args.input_channels,
                        fc1_dim=args.fc1_dim,
                        fc2_dim=args.fc2_dim,
                        out_dim=args.out_dim,
                        num_maps=args.num_maps,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        train_path=args.train_path,
                        test_path=args.test_path,
                        model_name=args.model_name)

        pp.train()

if __name__ == '__main__':
    main()
