# PreVIousNets

CNN models used in *PreVIous: A Methodology for Prediction of Visual Inference Performance on IoT Devices*.

## Files
* **Scripts.**
  * Python file for generating model definitions. Parameters include input dimensions and network architecture.
* **Models.** Network definitions in Caffe format.
  * PreVIousNet01: Input resolutions (HxW) are adjustable by editing corresponding lines `input_dim: H`. For adjusting input channel (C), use the model generation script.
  * PreVIousNet02: Number of input channels is adjustbale by editing corresponding line `input_dim: C`.
* **Docs.** 
  * Visualization figures.

## Citation
D. Velasco-Montero, *et. al.*, PreVIous: A Methodology for Prediction of Visual Inference Performance on IoT Devices. 
