# -*- coding: utf-8 -*-
"""
@author: D. Velasco-Montero

"""

# Partially based on:
# https://github.com/Cysu/pre-resnet-gen-caffe/blob/master/net_def.py

import os
import sys
import google.protobuf as pb
from argparse import ArgumentParser

CAFFE_ROOT = ''  # set path to CAFFE
if os.path.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe 
from caffe.proto import caffe_pb2

## FUNCTIONS FOR LAYER DEFINITION
## --------------------------------------------------------------------------
def CONV(bottom, num_output, kernel_size, stride, pad=0, group=1, bias_term=True, name='conv'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name   +'_'+str(kernel_size)+'_'+str(stride)+'_'+str(num_output)
    layer.type = 'Convolution'
    layer.bottom.extend([bottom])
    layer.top.extend([layer.name])
    layer.convolution_param.num_output = num_output
    layer.convolution_param.kernel_size.extend([kernel_size])
    layer.convolution_param.stride.extend([stride])
    layer.convolution_param.pad.extend([pad])
    layer.convolution_param.group = group
    layer.convolution_param.weight_filler.type = 'msra' # N(0,sigma^2). https://jihongju.github.io/2017/05/10/caffe-filler/
    layer.convolution_param.bias_term = bias_term
    if bias_term:
        layer.convolution_param.bias_filler.type = 'msra' # for testing purposes
    return layer

def POOL(bottom, kernel_size, stride, pad=0, pooling_method='max', name='pool'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + '_'+str(kernel_size)+'_'+str(stride)
    layer.type = 'Pooling'
    layer.bottom.extend([bottom])
    layer.top.extend([layer.name])
    if pooling_method != 'GAP': # Global Average Pooling
        layer.pooling_param.kernel_size = kernel_size
        layer.pooling_param.stride = stride
        layer.pooling_param.pad = pad   
        
    if pooling_method == 'max':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    elif pooling_method == 'ave':
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    elif pooling_method == 'GAP': # Global Average Pooling
        layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
        layer.pooling_param.global_pooling = True
    else:
        raise ValueError("Unknown Pooling method {}. Use 'max','ave','GAP'.".format(pooling_method))   
    return layer

def FC(bottom, num_output, bias_term=True, name='FC'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + '_'+str(num_output)
    layer.type = 'InnerProduct'
    layer.bottom.extend([bottom])
    layer.top.extend([layer.name])
    layer.inner_product_param.num_output = num_output
    layer.inner_product_param.weight_filler.type = 'msra'
    layer.inner_product_param.bias_term = bias_term
    if bias_term:
        layer.convolution_param.bias_filler.type = 'msra' # for testing purposes
    return layer

def BN(bottom, inplace=True, name=''):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + bottom+'_bn'
    layer.type = 'BatchNorm'
    layer.bottom.extend([bottom])
    layer.batch_norm_param.use_global_stats = True # For testing phase #http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html
    if inplace: layer.top.extend([bottom])
    else:       layer.top.extend([layer.name])
    return layer

def Scale(bottom, bias_term=True, inplace=True, name=''):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + bottom+'_scale'
    layer.type = 'Scale'
    layer.bottom.extend([bottom])
    layer.scale_param.bias_term = bias_term
    layer.scale_param.filler.type = 'msra'
    if bias_term:
        layer.scale_param.bias_filler.type = 'msra' # for testing purposes
    if inplace: layer.top.extend([bottom])
    else:       layer.top.extend([layer.name])
    return layer
  
def ReLU(bottom, inplace=True, name=''):
    layer = caffe_pb2.LayerParameter()
    layer.name = name + bottom+'_relu'
    layer.type = 'ReLU'
    layer.bottom.extend([bottom])
    if inplace: layer.top.extend([bottom])
    else:       layer.top.extend([layer.name])
    return layer

def Eltwise(bottoms, name='eltwise'): # ADD by default
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Eltwise'
    layer.bottom.extend(bottoms) # list of bottoms
    layer.top.extend([layer.name])
    return layer

def Concat(bottoms, name='concat'): 
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Concat'
    layer.bottom.extend(bottoms) # list of bottoms
    layer.top.extend([layer.name])
    return layer    

def Softmax(bottom, name='Prob'):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = 'Softmax'
    layer.bottom.extend([bottom])
    layer.top.extend([layer.name])
    return layer

## MODEL DEFINITION FROM A LIST OF LAYER DEFINITIONS
## --------------------------------------------------------------------------
def create_model(model_name, input_shape, version):
    
    model = caffe_pb2.NetParameter()
    model.name = model_name
    model.input.append('data')
    model.input_dim.extend(input_shape)
    
    layers = []
    
    if version == 'LeNet': # LeNet example of network file definition
        layers.append(CONV('data', 20,5,1, name='conv1'))
        layers.append(POOL(layers[-1].top[0],2,2, pooling_method='max', name='pool1'))
        layers.append(CONV(layers[-1].top[0], 50,5,1, name='conv2'))
        layers.append(POOL(layers[-1].top[0],2,2, pooling_method='max', name='pool2'))
        layers.append(FC(layers[-1].top[0], 500, name='ip1'))
        layers.append(ReLU(layers[-1].top[0], name='relu1'))
        layers.append(FC(layers[-1].top[0], 10, name='ip2'))
        layers.append(Softmax(layers[-1].top[0],name='prob')) 
            
    if version == '01':
        # Name's notation: Lx: level
        
        layers.append(BN('data'))               #0
        layers.append(Scale(layers[-1].top[0])) #1
        layers.append(ReLU(layers[-1].top[0]))  #2
        idx00 = len(layers)-1 # 2. L0 input index
        
        ### CONVOLUTIONS:    
        ## CONV (stacked): CHout > CHin
        layers.append(CONV(layers[2].top[0], input_shape[1]*2,1,1, name='convL0')) # 3 # Network backbone L0
        idx0 = len(layers)-1 # 3. L1 input index
        # + BN-scale-ReLU 
        layers.append(CONV(layers[-1].top[0], layers[-1].convolution_param.num_output*2,1,1, name='convL1')) #4 # Network backbone L1
        idx1 = len(layers)-1 # 4. L2 input index
        # + BN-scale-ReLU
         
        ## Special Convolution (branched): Dephtwise (DW) (group=CHin=CHout)
        layers.append(CONV(layers[idx00].top[0], input_shape[1],3,1, 
                           group=input_shape[1],pad=1,bias_term=False, name='convL0DW')) #5 
        # + Eltwise+Concat
        
        layers.append(CONV(layers[idx0].top[0], layers[idx0].convolution_param.num_output,3,1,
                           group=layers[idx0].convolution_param.num_output, pad=1,bias_term=False, name='convL1DW')) #6 
        # + Eltwise+Concat
        
        layers.append(CONV(layers[idx1].top[0], layers[idx1].convolution_param.num_output,3,1,
                           group=layers[idx1].convolution_param.num_output, pad=1,bias_term=False, name='convL2DW')) #7  # Network backbone L2
        idx2 = len(layers)-1 # 7. L2 output index
        # + Eltwise+Concat

        ## Special Convolutions (branched): Bottleneck/Squeeze (B) (k=s=1. CHout<CHin) 
        layers.append(CONV(layers[idx00].top[0], input_shape[1]/2,1,1, name='convL0B0')) #8 
        # + BN-scale-ReLU
        layers.append(CONV(layers[idx00].top[0], input_shape[1]/4,1,1, name='convL0B1')) 
        # + BN-scale-ReLU
        layers.append(CONV(layers[idx0].top[0], input_shape[1]/2,1,1, name='convL1B0'))
        # + Eltwise+Concat
        layers.append(CONV(layers[idx0].top[0], input_shape[1]/4,1,1, name='convL1B1')) 
        # + Eltwise+Concat
        layers.append(CONV(layers[idx1].top[0], input_shape[1]/2,1,1, name='convL2B0'))
        layers.append(CONV(layers[idx1].top[0], input_shape[1]/4,1,1, name='convL2B1'))  #13
        
        ## Other k,s in Convolutions (branched) CHout > CHin
        layers.append(CONV(layers[idx00].top[0], input_shape[1]*2,3,1, name='convL0')) #14
        layers.append(CONV(layers[idx00].top[0], input_shape[1]*2,3,2, name='convL0')) #15
        # + BN-scale-ReLU
        
        layers.append(CONV(layers[idx0].top[0], layers[idx0].convolution_param.num_output*2,3,1, name='convL1')) 
        layers.append(CONV(layers[idx0].top[0], layers[idx0].convolution_param.num_output*2,3,2, name='convL1')) #17
        lenL = len(layers)-1
        # + BN-scale-ReLU
        
        
        ### ACTIVATION LAYERS: BN, Scale, ReLU:
        # Inserted between some previously defined CONV layers
        addidx = 0 # idx shift since new layers are inserted
        for lay in [idx0, idx1, idx2+1, idx2+2, lenL-2, lenL]: # stacked to: L0, L1, L0B0, L0B1, L0(s=2), L1(s=2)
            l = lay +addidx
            layers.insert(l+1, BN(layers[l].top[0], inplace=True)) #18
            layers.insert(l+2, Scale(layers[l+1].top[0]))
            layers.insert(l+3, ReLU(layers[l+2].top[0], inplace=True))  
            addidx += 3
         # All previous layers indexes are now displaced
        idx1 += 3; idx2 += 6

        ### POOL
        # Inputs from: L0 and L2 input
        layers.append(POOL(layers[idx00].top[0], 3,2, pooling_method='max', name='poolL0'))
        layers.append(POOL(layers[idx00].top[0], 2,1, pooling_method='max', name='poolL0'))
        layers.append(POOL(layers[idx00].top[0], 3,2, pooling_method='GAP', name='GAPL0'))
        
        layers.append(POOL(layers[idx1].top[0], 3,2, pooling_method='max', name='poolL2'))
        layers.append(POOL(layers[idx1].top[0], 2,1, pooling_method='max', name='poolL2'))
        layers.append(POOL(layers[idx1].top[0], 3,2, pooling_method='GAP', name='GAPL2'))
        
        ### MULTIPLE-INPUT LAYERS: ELTWISE AND CONCAT
        # taking multiple previous outputs as inputs        
        layers.insert(idx1+5, Eltwise([layers[idx00].top[0], layers[idx1+4].top[0]], name='eltwiseL0') ) # inputs: network input & CONV L0(k=s=1)
        layers.insert(idx1+6, Concat([layers[idx00].top[0],  layers[idx1+4].top[0]], name='concatL0') )
    
        layers.insert(idx1+8, Eltwise([layers[idx0].top[0], layers[idx1+7].top[0]], name='eltwiseL1') ) # inputs: L1 input & CONV L1(k=s=1)
        layers.insert(idx1+9, Concat([layers[idx0].top[0],  layers[idx1+7].top[0]], name='concatL1') )
        idx2 += 4
        layers.insert(idx2+1, Eltwise([layers[idx1].top[0], layers[idx2].top[0]], name='eltwiseL2') ) # inputs: L2 input & CONV L2(k=s=1)
        layers.insert(idx2+2, Concat([layers[idx1].top[0],  layers[idx2].top[0]], name='concatL2') ) 

        layers.insert(idx2+12, Eltwise([layers[idx2+3].top[0], layers[idx2+11].top[0]], name='eltwiseL01b') ) # inputs: CONV L0 B0 & CONV L1 B0
        layers.insert(idx2+13, Concat([layers[idx2+3].top[0],  layers[idx2+11].top[0]], name='concatL01b') ) 
        layers.insert(idx2+15, Eltwise([layers[idx2+7].top[0], layers[idx2+14].top[0]], name='eltwiseL01c') ) # inputs: CONV L0 B1 & CONV L1 B1
        layers.insert(idx2+16, Concat([layers[idx2+7].top[0],  layers[idx2+14].top[0]], name='concatL01c') ) 
        
      
    if version == '02':
        # First cluster of FC layers:
        layers.append(FC('data', input_shape[1]*2, name='FCl1'))     #0
        layers.append(FC('data', input_shape[1]*4, name='FCl1'))     #1
        layers.append(FC('data', input_shape[1]*8, name='FCl1'))     #2
        layers.append(FC('data', input_shape[1]*16, name='FCl1'))    #3
        layers.append(FC('data', 10, name='FCl1'))                   #4  -Customized value C1
        layers.append(FC('data', 1000, name='FCl1'))                 #5  -Customized value C2
        layers.append(Softmax(layers[4].top[0],name='Softmax_10'))
        layers.append(Softmax(layers[5].top[0],name='Softmax_1000')) 
        
        for k in range(4): # Other clusters:
            layers.append(FC(layers[k].top[0], layers[0].inner_product_param.num_output*2, name='FCl2_'+str(k)))
            layers.append(FC(layers[k].top[0], layers[0].inner_product_param.num_output*4, name='FCl2_'+str(k)))
            layers.append(FC(layers[k].top[0], layers[0].inner_product_param.num_output*8, name='FCl2_'+str(k)))
            #layers.append(FC(layers[k].top[0], layers[0].inner_product_param.num_output*32, name='FCl2_'+str(k)))
            layers.append(FC(layers[k].top[0], 10, name='FCl2_'+str(k)))
            layers.append(FC(layers[k].top[0], 1000, name='FCl2_'+str(k)))
            
            layers.append(Softmax(layers[k].top[0],name='Softmax'+str(k)))    

        # Additional Softmax layers:
        for k in [2, 50, 100, 200, 500, 2000]: #-Customized values C3-C8
        #for k in np.linspace(50,2000, 6): # 6 = number of extra Softmax layers
            n = int(k)
            layers.append(FC(layers[4].top[0],n, name='FC_to'))  # 'bridge' layers
            layers.append(Softmax(layers[-1].top[0],name='Softmax_'+str(n)))    
                   
    model.layer.extend(layers)
    #print(pb.text_format.MessageToString(model) )
    return model


def main(args):
    model = create_model(args.name, args.shape, args.version)
    
    # Save model definition in file .prototxt
    filename = os.path.join(os.path.dirname(__file__), 
                            args.file + args.name + "_{:03d}".format(args.shape[1]) +'.prototxt' )

    with open(filename, 'w') as f:
        f.write(pb.text_format.MessageToString(model))    
    print('Saved Model File in ' + filename)
    
    # Save randomly-initialized weights in binary file .caffemodel
    #filename_w = os.path.join(os.path.dirname(__file__), 
    #                        args.file+ args.name + "_{:03d}".format(args.shape[1]) +'.caffemodel' )

    #net = caffe.Net(filename.encode("utf-8"), caffe.TEST)
    #net.save(filename_w.encode("utf-8"))
    #print('Saved Weights File in ' + filename_w)    
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n','--name', type=str, default='NET')
    parser.add_argument('-s', '--shape', type=int, nargs=4, default=[1,64,28,28])
    parser.add_argument('-f', '--file', type=str, default='')
    parser.add_argument('-v', '--version', type=str, choices=['01','02','LeNet'],default='01')
    args = parser.parse_args()
    main(args)
