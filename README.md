# PreVIousNet

CNN models used in *PreVIous: A Methodology for Prediction of Visual Inference Performance on IoT Devices*.

## Files
* **Scripts.**
  * create_prototxt.py: Python file for generating model definitions in Caffe format. Parameters include input dimensions (batch, channels, height, widht) and network architecture (01, 02). 
     
     Use example: `create_prototxt.py --name PreVIousNet01 -v 01 -s 1 64 28 28`
* **Models.** Generated network definitions in Caffe format. Input size (H,W,C) can be further adjusted by using the generator script with corresponding arguments.  
  * PreVIousNet01.prototxt. If only a change in input resolution (HxW) is desired, just edit corresponding lines 5-6 `input_dim: H/W`. For adjusting number of input channels (C), use the model generator script.
  * PreVIousNet02.prototxt: For adjusting number of input channels (C), use the model generator script.
* **Docs.** 
  * Visualization figure of the network.

## Citation
D. Velasco-Montero, *et. al.*, PreVIous: A Methodology for Prediction of Visual Inference Performance on IoT Devices. 
