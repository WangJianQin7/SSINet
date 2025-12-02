# SSINet

# Network Architecture

# Requirements
python 3.7 + pytorch 1.9.0 + imageio 2.22.2
# Saliency maps
We provide saliency maps of our SSINet on ORSSD and EORSSD datasets.  
[SSINet](https://pan.baidu.com/s/1PV8U5FDI4iKWkqknIv8LFQ) (code:SSIN)
# Training
Run train_SSINet.py.  
For SSINet.py, please download the pre-trained weights for the backbone network [MobileNet V3](https://pan.baidu.com/s/15290CcRgzRXD-WKWyoajwQ) (code:SSIN) and put it in ./model/.
# Pre-trained model and testing
Download the following pre-trained model and put them in ./models/SSINet/, then run test_SAFINet.py.  
[SSINet_ORSSD](https://pan.baidu.com/s/1RQnxYR3kn-fUjmmVzQRrnQ) (code:SSIN)  
[SSINet_EORSSD](https://pan.baidu.com/s/1mL0CekNATMHzWkVGlI86xw) (code:SSIN)
# Evaluation Tool
You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.
