<h2>EfficientNet-Peripheral-Blood-Cell (Updated: 2023/03/26)</h2>
<a href="#1">1 EfficientNetV2 Peripheral-Blood-Cell Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Peripheral Blood Cell dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Peripheral Blood Cell Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Peripheral-Blood-Cell Classification</a>
</h2>

 This is an experimental Peripheral-Blood-Cell Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>

The orignal Peripheral Blood Cell image dataset used here has been taken from the following web site:

https://data.mendeley.com/datasets/snkd93bnjr/1

A dataset for microscopic peripheral blood cell images for development 
of automatic recognition systems

<pre>
Acevedo, Andrea; Merino, Anna; Alférez, Santiago; Molina, Ángel; Boldú, Laura; Rodellar, José (2020), 
“A dataset for microscopic peripheral blood cell images for development of automatic recognition 
systems”, 
Mendeley Data, V1, doi: 10.17632/snkd93bnjr.1
</pre>
<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/EfficientNet-Peripheral_Blood_Cell.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Peripheral_Blood_Cell
        ├─eval
        ├─evaluation
        ├─inference        
        └─test
</pre>
<h3>
<a id="1.2">1.2 Peripheral-Blood-Cell dataset</a>
</h3>

Please download the dataset <b>PBC_dataset_normal_DIB.zip</b> from the following web site:
<br>
<a href="https://data.mendeley.com/datasets/snkd93bnjr/1">
A dataset for microscopic peripheral blood cell images for development of automatic recognition systems</a>
<br>
<br>
The original dataset contains the following eight classes (types) of Blood Cell:<br>
<pre>
basophil
eosinophil
erythroblast
ig
lymphocyte
monocyte
neutrophil
platelet
</pre>

We have splitted the image files in the original PBC_dataset_normal_DIB folder to train and test dataset by using 
<a href="./projects/Peripheral_Blood_Cell/split_master.py">split_master.py</a>
script.<br>
<pre>
>python split_master.py
</pre>

You can download the new Peripheral_Blood_Cell dataset splitted to train and test from 
<a href="https://drive.google.com/file/d/12nJmOmDJHOZ5U7GXQOjRlZpSw1EgNYSS/view?usp=sharing">here</a>,
<br>

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Peripheral_Blood_Cell
        ├─Peripheral_Blood_Cell
        │  ├─test
        │  │  ├─basophil
        │  │  ├─eosinophil
        │  │  ├─erythroblast
        │  │  ├─ig
        │  │  ├─lymphocyte
        │  │  ├─monocyte
        │  │  ├─neutrophil        
        │  │  └─platelet
        │  └─train
        │      ├─basophil
        │      ├─eosinophil
        │      ├─erythroblast
        │      ├─ig
        │      ├─lymphocyte
        │      ├─monocyte
        │      ├─neutrophil
        │      └─platelet
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        └─test
　...
</pre>

<br>
The number of images in train and test dataset:<br>
<img src="./projects/Peripheral_Blood_Cell/_Peripheral_Blood_Cell_.png" width="640" height="auto">
<br>

Sample images of Peripheral_Blood_Cell/train/basophil:<br>
<img src="./asset/sample_train_images_basophil.png" width="840" height="auto">
<br> 

Sample images of Peripheral_Blood_Cell/train/eosinophil:<br>
<img src="./asset/sample_train_images_eosinophil.png" width="840" height="auto">
<br> 

Sample images of Peripheral_Blood_Cell/train/erythroblast:<br>
<img src="./asset/sample_train_images_erythroblast.png" width="840" height="auto">
<br> 

Sample images of Peripheral_Blood_Cell/train/ig:<br>
<img src="./asset/sample_train_images_ig.png" width="840" height="auto">
<br> 

Sample images of Peripheral_Blood_Cell/train/lymphocyte:<br>
<img src="./asset/sample_train_images_lymphocyte.png" width="840" height="auto">
<br> 

Sample images of Peripheral_Blood_Cell/train/monocyte:<br>
<img src="./asset/sample_train_images_monocyte.png" width="840" height="auto">
<br> 


Sample images of Peripheral_Blood_Cell/train/neutrophil:<br>
<img src="./asset/sample_train_images_neutrophil.png" width="840" height="auto">
<br> 

Sample images of Peripheral_Blood_Cell/train/platelet:<br>
<img src="./asset/sample_train_images_platelet.png" width="840" height="auto">
<br> 


<br>


<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Peripheral-Blood-CellClassification</a>
</h2>
We have defined the following python classes to implement our Peripheral-Blood-CellClassification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Peripheral-Blood-Cell Model.
Please download the pretrained checkpoint file 
from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
        ├─Peripheral_Blood_Cell
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Peripheral-Blood-Cellefficientnetv2 model by using
<b>Peripheral_Blood_Cell/train</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-m ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=360 ^
  --eval_image_size=360 ^
  --data_dir=./Peripheral_Blood_Cell/train ^
  --data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.5 ^
  --num_epochs=100 ^
  --batch_size=8 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=False
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 8
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.4, 1.6]
data_format        = "channels_last"

</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Peripheral_Blood_Cell/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Peripheral_Blood_Cell/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/train_at_epoch_27_0325-2.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Peripheral_Blood_Cell/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Peripheral_Blood_Cell/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the skin cancer lesions in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.5 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=360 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
basophil
eosinophil
erythroblast
ig
lymphocyte
monocyte
neutrophil
platelet
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Peripheral_Blood_Cell/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Peripheral_Blood_Cell/Peripheral_Blood_Cell/test">Peripheral_Blood_Cell/test</a>.
<br>
<img src="./asset/test.png" width="840" height="auto"><br>


<br>
<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Peripheral_Blood_Cell/inference/inference.csv">inference result file</a>.
<br>At this time, you can see the inference accuracy for the test dataset by our trained model is very low.
More experiments will be needed to improve accuracy.<br>

<br>
Inference console output:<br>
<img src="./asset/inference_at_epoch_27_0325-2.png" width="740" height="auto"><br>
<br>

Inference result (<a href="./projects/Peripheral_Blood_Cell/inference/inference.csv">inference.csv</a>):<br>
<img src="./asset/inference_at_epoch_27_0325-2_csv.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Peripheral_Blood_Cell/Peripheral_Blood_Cell/test">
Peripheral_Blood_Cell/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Peripheral_Blood_Cell/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.5 ^
  --eval_image_size=360 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Peripheral_Blood_Cell/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Peripheral_Blood_Cell/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/evaluate_at_epoch_27_0325-2.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/classification_report_at_epoch_27_0325-2.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Peripheral_Blood_Cell/evaluation/confusion_matrix.png" width="740" height="auto"><br>

<br>
<h3>
References
</h3>
<b>1. A dataset for microscopic peripheral blood cell images for development of automatic recognition systems</b><br>
<pre>
https://data.mendeley.com/datasets/snkd93bnjr/1
</pre>

<b>2. Acute Myeloid Leukemia (AML) Detection Using AlexNet Model</b><br>
Maneela Shaheen,Rafiullah Khan, R. R. Biswal, Mohib Ullah,1Atif Khan, M.Irfan Uddin,4Mahdi <br>
Zareei and Abdul Waheed

<pre>
https://www.hindawi.com/journals/complexity/2021/6658192/
</pre>

<b>3. BCNet: A Deep Learning Computer-Aided Diagnosis Framework for Human Peripheral Blood Cell Identification</b><br>
Channabasava Chola, Abdullah Y. Muaad, Md Belal Bin Heyat, J. V. Bibal Benifa, Wadeea R. Naji, K. Hemachandran, <br> 
Noha F. Mahmoud, Nagwan Abdel Samee, Mugahed A. Al-Antari, Yasser M. Kadah and Tae-Seong Kim<br>
<pre>
https://www.mdpi.com/2075-4418/12/11/2815
</pre>

<b>4. Deep CNNs for Peripheral Blood Cell Classification</b><br>
Ekta Gavas and Kaustubh Olpadkar <br>
<pre>
https://arxiv.org/pdf/2110.09508.pdf
</pre>

