import os
import argparse
from torch.backends import cudnn
from data_loader import get_loader
import wandb
from solver import Solver

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             input_path=args.saved_path_train,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train(args)
    elif args.mode == 'test':
        solver.test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--net', type=str, default='VGG')
    parser.add_argument('--fine', type=str, default=False)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--load_mode', type=int, default=0)
    #parser.add_argument('--data_path', type=str, default=r'D:\Mayo\LDCT-and-Projection-data')
    parser.add_argument('--saved_path_train', type=str, default=r'D:\TG\RGB\train')
    #parser.add_argument('--saved_path_val', type=str, default=r'D:\Mayo\npy_img_val')
    #parser.add_argument('--saved_path_test', type=str, default=r'D:\Mayo\npy_img_test')
    parser.add_argument('--save_path', type=str, default=r'D:\TG\ckpts')

    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--save_iters', type=int, default=1000)
    parser.add_argument('--test_iters', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)
