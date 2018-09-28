#! /usr/bin/env python
# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, Thu Sep 20 21:57:32 CST 2018
# ***
# ************************************************************************************/


import sys
from PIL import Image
import torch

import model
# import graph


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        encoder = model.encoder_load("models/encoder.pth")
        encoder.eval()
        # graph.show(encoder)

        encoder.to(device)

        decoder = model.decoder_load("models/decoder.pth")
        # graph.show(decoder)

        decoder.eval()
        decoder.to(device)

        img = Image.open(sys.argv[1])
        # x -> z -> y
        x = model.image_to_tensor(img).to(device)
        z = encoder(x)
        y = decoder(z)

        y = torch.Tensor(y.cpu())
        y.clamp_(0, 1)
        img = model.image_from_tensor(y)
        img.show()
