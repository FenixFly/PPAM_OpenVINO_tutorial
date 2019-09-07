import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from itertools import product
from openvino.inference_engine import IENetwork, IECore

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to a folder with \
        images or path to an image files', required=True, type=str, nargs='+')
    parser.add_argument('-o', '--output_folder', help='Path to an output folder',
        type=str, default='')
    parser.add_argument('-l', '--cpu_extension', help='MKLDNN \
        (CPU)-targeted custom layers.Absolute path to a shared library \
        with the kernels implementation', type=str, default=None)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-em', '--execution_mode', help='Execution mode: \
        sync or async', type=str, default='sync')
    parser.add_argument('-c','--classes', help='File containing classes names', type=str,
        default=None)
    return parser

def prepare_model(model, weights, cpu_extension, device, log):

    log.info('Creating IECore object for inference')
    
    # Remove next line
    net = None; ie = None
    #
    # Insert your code here
    #

    return net, ie


def get_image_list(input, log):
    log.info('Creating list of input images')
    extensions = tuple({'.jpg', '.png', '.gif', '.bmp', '.JPEG'})
    
    # Remove next line
    data = None
    #
    # Insert your code here
    #
    
    return data

def convert_images(net, data, log):
    log.info('Reading images and converting to input tensor')
    n, c, h, w = net.inputs[next(iter(net.inputs))].shape
    
    # Remove next line
    images = None
    #
    # Insert your code here
    #
    
    return images



def infer_sync(images, exec_net, net):
    log.info('Start sync execution mode')
    
    # Remove next line
    results = None
    #
    # Insert your code here
    #
    
    return results

def infer_async(images, exec_net, net):
    log.info('Start async execution mode')
    
    # Remove next line
    results = None
    #
    # Insert your code here
    #
    
    return results

def detection_output(input_data, output_folder, pred, classes_file, log):
    classes = [line.rstrip('\n') for line in open(classes_file)]
    log.info('Execute detection output processing')
    
    #
    # Insert your code here
    #
    
    return

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s',
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    #
    # Insert your code here
    #
    
    
    # Prepare model
    net, ie = prepare_model(args.model, args.weights, args.cpu_extension, 
                            args.device, log)
    # Get image list
    
    # Convert images 
    
    # Load network to core
    
    # Run inference
    
    # Detection output
    
    # Free memory 
    del net
    del ie
        
    return 0
    
if __name__ == '__main__':
    sys.exit(main())
