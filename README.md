# Attention is All We Need: Nailing Down Object-centric Attention for Egocentric Activity Recognition

The git contains the source code associated with our BMVC 2018 paper:
"Attention is All We Need: Nailing Down Object-centric Attention for Egocentric Activity Recognition"
The paper is available in [here](https://arxiv.org/pdf/1807.11794.pdf).

#### Prerequisites

* Python 3.5
* Pytorch 0.3.1
  #### 

#### Running

* ##### RGB

  * ###### Stage 1
  * ```
    python main-run-rgb.py --dataset gtea_61 
    --stage 1 
    --trainDatasetDir ./dataset/gtea_61/split2/train 
    --outDir experiments 
    --seqLen 25 
    --trainBatchSize 32 
    --numEpochs 300 
    --lr 1e-3 
    --stepSize 25 75 150 
    --decayRate 0.1 
    --memSize 512
    ```
  * ###### Stage 2
  * ```
    python main-run-rgb.py --dataset gtea61 
    --stage 2 
    --trainDatasetDir ./dataset/gtea_61/split2/train 
    --outDir experiments 
    --stage1Dict best_model_state_dict.pth 
    --seqLen 25 
    --trainBatchSize 32 
    --numEpochs 150 
    --lr 1e-4 
    --stepSize 25 75 
    --decayRate 0.1 
    --memSize 512
    ```
* ##### **Flow**
* ```
  python main-run-flow.py --dataset gtea61 
  --trainDatasetDir ./dataset/gtea_61/split2/train 
  --outDir experiments 
  --stackSize 5 
  --trainBatchSize 32 
  --numEpochs 750 
  --lr 1e-2 
  --stepSize 150 300 500 
  --decayRate 0.5
  ```
* ##### **Two Stream**
* ```
  python main-run-twoStream.py --dataset gtea61 
  --flowModel ./models/best_model_state_dict_flow_split2.pth 
  --rgbModel ./models/best_model_state_dict_rgb_split2.pth 
  --trainDatasetDir ./dataset/gtea_61/split2/train 
  --outDir experiments 
  --seqLen 25 
  --stackSize 5 
  --trainBatchSize 32 
  --numEpochs 250 
  --lr 1e-2 
  --stepSize 1 
  --decayRate 0.99 
  --memSize 512
  ```

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



