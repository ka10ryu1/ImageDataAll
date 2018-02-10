#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import cv2
import argparse
import numpy as np
from pathlib import Path

import imgfunc as IMG
import func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('color',
                        help='使用する入力画像のあるフォルダ')
    parser.add_argument('mono',
                        help='使用する正解画像のあるフォルダ')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--round', '-r', type=int, default=100,
                        help='切り捨てる数（default: 100）')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合（default: 0.9）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def readImages(folder, channel, ext):
    # OpenCV形式で画像を読み込むために
    # チャンネル数をOpenCVのフラグ形式に変換する
    ch = IMG.getCh(channel)
    # OpenCV形式で画像をリストで読み込む
    print('read images...')
    img_path = [i.as_posix() for i in list(Path(folder).glob('*' + ext))]
    img_path.sort()
    imgs = [cv2.imread(name, ch) for name in img_path]
    # 画像を分割する（正解データに相当）
    return IMG.split(
        IMG.rotate(imgs), args.img_size,
        args.round, flg=cv2.BORDER_REFLECT_101
    )


def main(args):

    # 入力のカラー画像を読み込む
    x = readImages(args.color, 3, '.JPG')
    # 正解のモノクロ画像を読み込む
    y = readImages(args.mono, 1, '.png')

    # 画像の並び順をシャッフルするための配列を作成する
    # colorとmonoの対応を崩さないようにシャッフルしなければならない
    # また、train_sizeで端数を切り捨てる
    print('shuffle images...')
    shuffle = np.random.permutation(range(len(x)))
    train_size = int(len(x) * args.train_per_all)
    train_x = x[shuffle[:train_size]]
    train_y = y[shuffle[:train_size]]
    test_x = x[shuffle[train_size:]]
    test_y = y[shuffle[train_size:]]
    print(
        'train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    print('save npz...')
    size_str = '_' + str(args.img_size).zfill(2) + 'x' + \
        str(args.img_size).zfill(2)
    num_str = '_' + str(train_x.shape[0]).zfill(6)
    np.savez(F.getFilePath(args.out_path, 'train' + size_str + num_str),
             x=train_x, y=train_y)
    num_str = '_' + str(test_x.shape[0]).zfill(6)
    np.savez(F.getFilePath(args.out_path, 'test' + size_str + num_str),
             x=test_x, y=test_y)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
