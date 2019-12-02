# PreVIousNets

CNN models used in *PreVIous: A Methodology for Prediction of Visual Inference Performance on IoT Devices*.

## Files
* **Scripts.**
  * create_prototxt.py: Python file for generating model definitions. Parameters include input dimensions and network architecture. 
     
     Use example: `create_prototxt.py --name PreVIousNet01 -v 01 -s 1 64 28 28`
* **Models.** Network definitions in Caffe format.
  * PreVIousNet01: Input resolutions (HxW) are adjustable by editing corresponding lines 5-6 `input_dim: H/W`. For adjusting input channels (C), use the model generation script with corresponding arguments.
  * PreVIousNet02: For adjusting number of input channels (C), use the model generation script with corresponding arguments.
* **Docs.** 
  * Visualization figures.

## Citation
D. Velasco-Montero, *et. al.*, PreVIous: A Methodology for Prediction of Visual Inference Performance on IoT Devices. 
