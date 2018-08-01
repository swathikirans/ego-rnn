# Attention is All We Need: Nailing Down Object-centric Attention for Egocentric Activity Recognition

The git contains the source code associated with our BMVC 2018 paper:
"Attention is All We Need: Nailing Down Object-centric Attention for Egocentric Activity Recognition"
The paper is available in [here](https://arxiv.org/pdf/1807.11794.pdf).

#### Prerequisites

* Python 3.5
* Pytorch 0.3.1
  #### 

*Training code will be released soon!*

#### **Evaluating the models**

* ##### **RGB**
* ```
  python eval-run-rgb.py --dataset gtea61 
  --datasetDir ./dataset/gtea_61/split2/test 
  --modelStateDict best_model_state_rgb.pth 
  --seqLen 25 
  --memSize 512
  ```
* ##### **Flow**
* ```
  python eval-run-rgb.py --dataset gtea61 
  --datasetDir ./dataset/gtea_61/split2/test 
  --modelStateDict best_model_state_flow.pth 
  --stackSize 5 
  --numSegs 5
  ```
* ##### **Two Stream**
* ```
  python eval-run-twoStream-joint.py --dataset gtea61 
  --datasetDir ./dataset/gtea_61/split2/test 
  --modelStateDict best_model_state_twoStream.pth 
  --seqLen 25 
  --stackSize 5 
  --memSize 512
  ```

#### **Pretrained models**

The models trained on the fixed split \(S2\) of GTEA 61 can be downloaded from the following links

* RGB model [https://drive.google.com/open?id=1B7Xh6hQ9Py8fmL-pjmLzlCent6dnuex5](https://drive.google.com/open?id=1B7Xh6hQ9Py8fmL-pjmLzlCent6dnuex5 "RGB model")
* Flow model [https://drive.google.com/open?id=1eG-ZF1IwOtYJqpIIeMASURB0uyCM\_cFd](https://drive.google.com/open?id=1eG-ZF1IwOtYJqpIIeMASURB0uyCM_cFd "Flow model")
* Two stream model [https://drive.google.com/open?id=11U5xbrOr8GtEhpkxY2lpPsyFDFJ8savp](https://drive.google.com/open?id=11U5xbrOr8GtEhpkxY2lpPsyFDFJ8savp "Two stream model")

The dataset can be downloaded from the following link:

[http://www.cbi.gatech.edu/fpv/](http://www.cbi.gatech.edu/fpv/)

Once the videos are downloaded, extract the frames and optical flow using the following implementation:

[https://github.com/yjxiong/dense\_flow](https://github.com/yjxiong/dense_flow)

Run 'prepareGTEA61Dataset.py' script to make the dataset.

Alternatively, the frames and the corresponding warp optical flow of the GTEA 61 dataset can be downloaded from the following link

* [https://drive.google.com/file/d/1\_y8Y3PnCXsngmZVMqZbg-AfJyIdOeQ2\_/view?usp=sharing](https://drive.google.com/file/d/1_y8Y3PnCXsngmZVMqZbg-AfJyIdOeQ2_/view?usp=sharing "GTEA61")



