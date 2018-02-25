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
    parser.add_argument('color',   help='使用する入力画像')
    parser.add_argument('duotone', help='使用する正解画像')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ（default: 32 pixel）')
    parser.add_argument('--round', '-r', type=int, default=100,
                        help='切り捨てる数（default: 100）')
    # parser.add_argument('--channel', '-c', type=int, default=1,
    #                     help='入力画像のチャンネル数（default: 1）')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合（default: 0.9）')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def saveNPZ(x, y, name, folder, size):
    size_str = '_' + str(size).zfill(2) + 'x' + str(size).zfill(2)
    num_str = '_' + str(x.shape[0]).zfill(6)
    np.savez(F.getFilePath(folder, name + size_str + num_str), x=x, y=y)


def main(args):

    # 入力のカラー画像を読み込む
    x = cv2.imread(args.color, IMG.getCh(3))
    # 正解のモノクロ画像を読み込む
    y = cv2.imread(args.duotone, IMG.getCh(1))

    x, _ = IMG.split(IMG.rotate([x]), args.img_size, args.round)
    y, _ = IMG.split(IMG.rotate([y]), args.img_size, args.round)

    # 画像の並び順をシャッフルするための配列を作成する
    # colorとmonoの対応を崩さないようにシャッフルしなければならない
    # また、train_sizeで端数を切り捨てる
    print('shuffle images...')
    shuffle = np.random.permutation(range(len(x)))
    train_size = int(len(x) * args.train_per_all)
    print(train_size, len(x))  # , x.shape)
    train_x = IMG.imgs2arr(x[shuffle[:train_size]], dtype=np.float16)
    train_y = IMG.imgs2arr(y[shuffle[:train_size]], dtype=np.float16)
    test_x = IMG.imgs2arr(x[shuffle[train_size:]], dtype=np.float16)
    test_y = IMG.imgs2arr(y[shuffle[train_size:]], dtype=np.float16)
    print('train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    print('save npz...')
    saveNPZ(train_x, train_y, 'train', args.out_path, args.img_size)
    saveNPZ(test_x, test_y, 'test', args.out_path, args.img_size)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
