# coding=utf-8
import os
import argparse
from math import *
import numpy as np
import cv2
# import PIL
from PIL import Image, ImageFont, ImageDraw
from lib.common import *


class GenPlate:
    def __init__(self, font_cn, font_en, bgs):
        self.fontC = ImageFont.truetype(font_cn, 43, 0)
        self.fontE = ImageFont.truetype(font_en, 60, 0)
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread("./plates/plate_1.bmp"), (226, 70))
        self.smu = cv2.imread("./smudginess/smu_2.jpg")
        self.bgs_path = []
        for parent, parent_folder, filenames in os.walk(bgs):
            for filename in filenames:
                path = parent + "/" + filename
                self.bgs_path.append(path)

    def draw(self, val):
        offset = 2
        self.img[0: 70, offset + 8:offset + 8 + 23] = gen_ch(self.fontC, val[0])
        self.img[0: 70, offset + 8 + 23 + 6:offset + 8 + 23 + 6 + 23] = gen_ch1(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0: 70, base: base + 23] = gen_ch1(self.fontE, val[i + 2])
        return self.img

    def generate(self, text):
        if len(text) == 7:
            # fg = self.draw(text.decode(encoding="utf-8"))
            fg = self.draw(text)
            fg = cv2.bitwise_not(fg)
            com = cv2.bitwise_or(fg, self.bg)
            # com = rot(com,r(60)-30,com.shape,30);
            com = rot(com, r(40) - 20, com.shape, 20)
            com = rot_random(com, 10, (com.shape[1], com.shape[0]))
            com = add_smu(com, self.smu)

            # com = tf_actor(com)
            com = random_environment(com, self.bgs_path)
            com = add_gauss(com, 1 + r(2))
            com = add_noise(com)
            return com

    @staticmethod
    def gen_plate_str(pos, val):
        plateStr = ""
        box = [0, 0, 0, 0, 0, 0, 0]
        if pos != -1:
            box[pos] = 1
        for unit, c_pos in zip(box, range(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if c_pos == 0:
                    plateStr += chars[r(31)]
                elif c_pos == 1:
                    plateStr += chars[41 + r(24)]
                else:
                    plateStr += chars[31 + r(34)]
        return plateStr

    def gen_batch(self, batch_size, pos, char_range, output_path, size):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i in range(batch_size):
            plateStr = self.gen_plate_str(-1, -1)
            print(plateStr)
            img = self.generate(plateStr)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, size)
            # filename = os.path.join(outputPath, str(i).zfill(4) + '.' + plateStr + ".jpg")
            filename = os.path.join(output_path, str(i).zfill(5) + '_' + plateStr + ".jpg")
            cv2.imwrite(filename, img)
            print(filename, plateStr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_ch', default='./font/platech.ttf')
    parser.add_argument('--font_en', default='./font/platechar.ttf')
    parser.add_argument('--bg_dir', default='./backgrounds')
    parser.add_argument('--out_dir', default='./input', help='output dir')
    parser.add_argument('--make_num', default=10, type=int, help='num')
    parser.add_argument('--img_w', default=120, type=int, help='num')
    parser.add_argument('--img_h', default=32, type=int, help='num')
    return parser.parse_args()


def main(args):
    g = GenPlate(args.font_ch, args.font_en, args.bg_dir)
    g.gen_batch(args.make_num, 2, range(31, 65), args.out_dir, (args.img_w, args.img_h))


if __name__ == '__main__':
    main(parse_args())
