#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, Thu Sep 20 21:42:14 CST 2018
# ***
# ************************************************************************************/


import argparse
import os
from os.path import basename
from os.path import splitext
import logging

import torch
from PIL import Image

from torchvision import transforms
from torchvision.utils import save_image

import model

parser = argparse.ArgumentParser()
parser.add_argument('-style', type=str, help='File path to the style image')
parser.add_argument(
    '-content', type=str, help='File path to the content image')
parser.add_argument(
    '-output',
    type=str,
    default='output',
    help='Directory to save the output images')

parser.add_argument('-e', '--encoder', type=str, default='models/encoder.pth')
parser.add_argument('-d', '--decoder', type=str, default='models/decoder.pth')

parser.add_argument(
    '-alpha',
    type=float,
    default=1.0,
    help='The weight that controls the degree of stylization. Should be [0, 1]'
)

if __name__ == '__main__':
    args = parser.parse_args()
    assert (args.content and args.style)

    if os.path.isdir(args.style):
        style_paths = [os.path.join(args.style, f) for f in os.listdir(args.style)]
    else:
        style_paths = [args.style]

    if os.path.isdir(args.content):
        content_paths = [
            os.path.join(args.content, f) for f in os.listdir(args.content)
        ]
    else:
        content_paths = [args.content]

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = model.encoder_load(args.encoder)
    encoder.eval()
    encoder.to(device)

    decoder = model.decoder_load(args.decoder)
    decoder.eval()
    decoder.to(device)

    T = transforms.Compose([
        transforms.ToTensor(),
    ])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    for content_path in content_paths:
        for style_path in style_paths:
            logging.info("Transfering from " + style_path + " to " + content_path +
                         " ...")

            content = T(Image.open(content_path).convert('RGB'))
            content = content.to(device).unsqueeze(0)

            style = T(Image.open(style_path).convert('RGB'))
            style = style.to(device).unsqueeze(0)

            with torch.no_grad():
                output = model.style_transfer(encoder, decoder, content, style,
                                              args.alpha)
            output = output.cpu()

            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                args.output,
                splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], ".jpg")

            save_image(output, output_name)
