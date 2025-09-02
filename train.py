import os
import glob
import shutil
import warnings
import argparse
from utils import *
from solver import Solver
from torch.backends import cudnn
from data_loader import get_loader


warnings.filterwarnings("ignore")

def read_image_file(str):
    files_image = glob.glob(str + '/*.tif')
    return files_image

def read_label_file(str):
    files_label = glob.glob(str + '/*.tif')
    return files_label

def main(config):

    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'MSU_Net', 'my_Net']:
        print(
             'ERROR!! model_type should be selected in U_Net/MSU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    config.train_result_path = os.path.join(config.train_result_path, config.model_type)
    if not os.path.exists(config.train_result_path):
        os.makedirs(config.train_result_path)

    for i in range(config.k):

        tv_image = read_image_file('/home/ubuntu/DeepAtlasmodel1111111/paper3/Dataset/hu260/test/')
        pet_image = read_image_file('/home/ubuntu/DeepAtlasmodel1111111/paper3/Dataset/hu260/pet_test/')
        tv_label = read_label_file('/home/ubuntu/DeepAtlasmodel1111111/paper3/Dataset/hu260/test_labels/')

        X_train, y_train, X_valid, y_valid, pet_valid, pet_train = k_fold_data(config.k, i, tv_image, tv_label, pet_image)

        print('*' * 25, '第', i + 1, '折', '*' * 25)

        config.model_path = ('/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/model1/%s/The%1d' % (config.model_type, i + 1))
        if os.path.exists(config.model_path):
            shutil.rmtree(config.model_path)  # 删除已存在的文件夹及其内容

        os.makedirs(config.model_path)
        print('Create path - %s' % config.model_path)

        config.train_result_path = ('/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/result1/train_result/%s/The%1d' % (config.model_type, i + 1))
        if os.path.exists(config.train_result_path):
            shutil.rmtree(config.train_result_path)  # 删除已存在的文件夹及其内容

        os.makedirs(config.train_result_path)
        print('Create path - %s' % config.train_result_path)

        config.val_result_path = ('/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/result1/val_result/%s/The%1d' % (config.model_type, i + 1))
        if os.path.exists(config.val_result_path):
            shutil.rmtree(config.val_result_path)  # 删除已存在的文件夹及其内容

        os.makedirs(config.val_result_path)
        print('Create path - %s \n' % config.val_result_path)

        lr = 0.01
        epoch = 1000

        config.num_epochs = epoch
        config.lr = lr

        print(config)

        train_loader = get_loader(imList=X_train,
                                  petlist=pet_train,
                                  labelList=y_train,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='train',
                                  drop_last=True)
        valid_loader = get_loader(imList=X_valid,
                                  petlist=pet_valid,
                                  labelList=y_valid,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  mode='valid',
                                  drop_last=True)
        test_loader = []

        solver = Solver(config, train_loader, valid_loader, test_loader)


        # Train and sample the images
        if config.mode == 'train':
            solver.train()
        elif config.mode == 'test':
            solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='train')

    # model_type and path
    parser.add_argument('--model_type', type=str, default='my_Net', help='U_Net/MSU_Net/my_Net')
    parser.add_argument('--model_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/model/')

    parser.add_argument('--train_result_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/result/train_result/')
    parser.add_argument('--val_result_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/result/val_result/')
    parser.add_argument('--test_result_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Diffnetv2/result/test_result/')

    parser.add_argument('--train_valid_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Dataset/hu260/train/')
    parser.add_argument('--pet_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Dataset/hu260/pet/')
    parser.add_argument('--test_path', type=str, default='/home/ubuntu/DeepAtlasmodel1111111/paper3/Dataset/hu260/test/')

    # hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs_test', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)  # 4
    parser.add_argument('--k', type=int, default=5)  # 5
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    config = parser.parse_args()
    # print(config)
    main(config)
