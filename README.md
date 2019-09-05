# PPAM OpenVINO tutorial

## To run sample, run ```object_detection.py``` with needed arguments.


python object_detection.py -i ../data/input -m ../models/ssd300.xml -w ../models/ssd300.bin -c ../models/pascal_voc_classes.txt -l "C:/Program Files (x86)/IntelSWTools/openvino/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll" -em async