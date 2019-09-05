import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
import copy
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
    model_xml = model
    model_bin = weights

    log.info('Creating IECore object for inference')
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, "CPU")

    log.info('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info('Checking layers are supported by IECore')
    if 'CPU' in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() \
            if l not in supported_layers]
        if not_supported_layers:
            log.error('Following layers are not supported by the IECore for specified' 
                'device {}:\n {}'.format(ie.device, ', '.join(not_supported_layers)))
            log.error('Please try to specify cpu extensions library path in sample\'s' 
                'command line parameters using -l or --cpu_extension command line'
                'argument')
            sys.exit(1)
    return net, ie


def get_image_list(input, log):
    extensions = tuple({'.jpg', '.png', '.gif', '.bmp', '.JPEG'})
    log.info('Creating list of input images')
    if os.path.isdir(input[0]):
        data = []
        for file in os.listdir(input[0]):
            if file.endswith(extensions):
                data.append(os.path.join(input[0], file))
    else:
        data = input
    return data

def convert_images(net, data, log):
    log.info('Getting shape of input tensor')
    n, c, h, w = net.inputs[next(iter(net.inputs))].shape
    log.info('Creating input images tensor')
    images = np.ndarray(shape=(len(data), c, h, w))
    for i in range(len(data)):
        image = cv2.imread(data[i])
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        images[i] = image
    return images

def infer_sync(images, exec_net, net):
    results = []
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    for image in images:
        res = exec_net.infer(inputs={input_blob : image})
        results.append(res[out_blob][0])
    return results

def infer_async(images, exec_net, net):
    number_iter = len(images)
    curr_request_id = 0
    prev_request_id = 1
    results = []
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    first = True
    
    for j in range(number_iter):
        exec_net.start_async(request_id = curr_request_id, 
                             inputs = {input_blob: images[j]})
        if first == True:
            first = False
            prev_request_id, curr_request_id = curr_request_id, prev_request_id
            continue
        
        if exec_net.requests[prev_request_id].wait(-1) == 0:
            results.append(copy.copy(exec_net.requests[prev_request_id].
                outputs[next(iter(net.outputs))][0]))
        prev_request_id, curr_request_id = curr_request_id, prev_request_id
    
    if exec_net.requests[prev_request_id].wait(-1) == 0:
        results.append(copy.copy(exec_net.requests[prev_request_id].
            outputs[next(iter(net.outputs))][0]))
        
    return results

def detection_output(input_data, output_folder, pred, classes_file, log):
    classes = [line.rstrip('\n') for line in open(classes_file)]
    threshold = 0.5

    for im_id, data in enumerate(pred):
        img_name = input_data[im_id]
        img = cv2.imread(img_name)
        h = img.shape[0]
        w = img.shape[1]
        for i in range(data.shape[1]):
            if data[0,i,2] > threshold:
                cv2.rectangle(img,
                          (int(data[0,i,3] * w), int(data[0,i,4] * h)),
                          (int(data[0,i,5] * w), int(data[0,i,6] * h)),
                          (0,255,0),1)
                cv2.putText(img, classes[int(data[0,i,1]-1)], 
                           (int(data[0,i,3] * w), int(data[0,i,6] * h)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 255), 2, cv2.LINE_AA)
        out_img = os.path.join(output_folder, os.path.basename(img_name))
        cv2.imwrite(out_img, img)
        log.info('Result image was saved to {}'.format(out_img))
    

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s',
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    try:
        net, ie = prepare_model(args.model, args.weights,
            args.cpu_extension, args.device, log)

        data = get_image_list(args.input, log)
        images = convert_images(net, data, log)

        log.info('Loading model to the core')
        if args.execution_mode == 'async':
            exec_net = ie.load_network(network=net, device_name=args.device, 
                                       num_requests = 2)
        else:
            exec_net = ie.load_network(network=net, device_name=args.device)

        log.info('Starting inference')
        if args.execution_mode == 'async':
            res = infer_async(images, exec_net, net)
        else:
            res = infer_sync(images, exec_net, net)

        log.info('Processing model output')
        detection_output(data, args.output_folder, res, args.classes, log)

        log.info('Free memory')
        del net
        del exec_net
        del ie
    except Exception as ex:
        log.error(str(ex))
        sys.exit(1)

if __name__ == '__main__':
    sys.exit(main())