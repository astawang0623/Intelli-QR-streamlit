import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename, splitext
from torchvision import transforms
from torchvision.utils import save_image
from . import transformer
from . import StyTR
import numpy as np


def test_transform(size, crop):
    transform_list = []
   
    # if size != 0: 
    #     transform_list.append(transforms.Resize(size))
    # if crop:
    #     transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    # print(type(size))
    transform_list = []    
    transform_list.append(transforms.Resize(size))
    # transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  
def get_parser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str,
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str,
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the output image(s)')
    parser.add_argument('--vgg_path', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='models/decoder.pth')
    parser.add_argument('--transform_path', type=str, default='models/transformer_iter_160000.pth')
    parser.add_argument('--embedding_path', type=str, default='models/embedding_iter_160000.pth')
    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)

    parser.add_argument('--style_interpolation_weights', type=str, default="")
    parser.add_argument('--alpha', type=float, default=1.0)
    return parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def style_transfer_main(args):
    content_size=args.content_size
    style_size=args.style_size
    crop='store_true'
    save_ext='.jpg'
    output_path=args.output_dir

    # Either --content or --content_dir should be given.
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --style_dir should be given.
    if args.style:
        style_paths = [Path(args.style)]    
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    if not os.path.exists(output_path):
        os.mkdir(output_path)


    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.transform_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    embedding_path = './nst_StyTr2/models/embedding_iter_160000.pth'
    # embedding_path = args.embedding_path
    new_state_dict = OrderedDict()
    # state_dict = torch.load(args.embedding_path)
    state_dict = torch.load(embedding_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
    network.eval()
    network.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    
    for content_path in content_paths:
        for style_path in style_paths:
            # print(content_path)
        
        
            content_tf1 = content_transform()       
            content = content_tf(Image.open(content_path).convert("RGB"))

            h,w,c=np.shape(content)    
            style_tf1 = style_transform(h,w)
            style = style_tf(Image.open(style_path).convert("RGB"))

        
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            
            with torch.no_grad():
                output= network(content,style)
            output = output.cpu()
            # result = tuple(o.cpu() for o in output)
                    
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_path, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
            if args.output_name:
                output_name = args.output_dir + "/" + args.output_name + save_ext
    
            save_image(output, output_name)
            # save_image(result, output_name)
    return output

def style_transfer_with_input(args, content_PIL, style_PIL):
    content_size=args.content_size
    style_size=args.style_size
    crop='store_true'

    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(args.vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.transform_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    embedding_path = './nst_StyTr2/models/embedding_iter_160000.pth'
    new_state_dict = OrderedDict()
    state_dict = torch.load(embedding_path)
    for k, v in state_dict.items():
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args)
    network.eval()
    network.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    
    content = content_tf(content_PIL)

    style = style_tf(style_PIL)


    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output= network(content,style)
    output = output.cpu()
            
    return output


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    style_transfer_main(args)
