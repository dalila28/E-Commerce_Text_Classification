# E-Commerce_Text_Classification 

### 1. Project Description
e-commerce is the platform where we can have information regarding latest market trends. By using any text documents that are available in the e-commerce platform we can synthesis beneficial information from there.Therefore, deep learning model such as Long Short Term Memory(LSTM) is used to make classification of texts. We never imagine that texts can be used to make prediction right? but with deep learning this can happen. The aim of this project is to make classification for texts in the dataset and categorize them into 'Household','Books', 'Clothing & Accesories' and 'Electronic'.

 * Challenges : when dealing with texts, we cannot avoid from preventing people only write words by using alphabets. For sure they will mix the sentences that they want to write with different kind of symbols. This is problem that i need to deal while handling the text dataset. In order to face this issue, one specific function is created to deal with symbols and unwanted strings in reviw_handler.py file.
 
### 2. Software used, framework,how to run project
   * Software needed:
     * Visual Studio Code as my IDE. You can get here https://code.visualstudio.com/download
     * Anaconda. https://www.anaconda.com/download

   * Framework:
     * I use Tensorflow and Keras framework to develop this deep learning project, as for Keras it already a part of TensorFlow’s core API.
   
   * How to run project:
     * Download project in the github
     * In Visual Studio Code make sure install Python
     * Open Anaconda prompt : "(OPTIONAL) IS FOR GPU INSTALLATION IF YOU NEED FOR CPU THEN IGNORE OPTIONAL"
        * (base) conda create -n "name u want to have" python=3.8
        * (env) conda install -c anaconda ipykernel
        * conda install numpy,conda install pandas,conda install matplotlib (run each of this one by one)
        * (OPTIONAL) conda install -c anaconda cudatoolkit=11.3
        * (OPTIONAL) conda install -c anaconda cudnn=8.2
        * (OPTIONAL) conda install -c nvidia cuda-nvcc
        * conda install git
        * 1 (a) create a folder named TensorFlow inside the tensorflow environment. For example: “C:\Users\< USERNAME >\Anaconda3\envs\tensorflow\TensorFlow”
        * (b) type: cd “C:\Users\<USERNAME>\Anaconda3\envs\tensorflow\TensorFlow” (to change directory to the newly created TensorFlow folder) 
        * (c) type: git clone https://github.com/tensorflow/models.git
        * conda install -c anaconda protobuf
        * 2 (a) type: cd “C:\Users\< USERNAME >\Anaconda3\envs\tensorflow\TensorFlow\models\research” (into TensorFlow\models\research for example)
        * b) type: protoc object_detection/protos/*.proto --python_out=.
        * 3 a) pip install pycocotools-windows
        * b) cp object_detection/packages/tf2/setup.py .
        * c) python -m pip install .
      * Test your installation (RESTART TERMINAL BEFORE TESTING)  
         * Inside C:\Users\< USERNAME > \Anaconda3\envs\tensorflow\TensorFlow\models\research
         * python object_detection/builders/model_builder_tf2_test.py The terminal should show OK if it passes all the tests
         
      * Open Visual Studio Code then open folder of downloaded file, search .py file then run it.
         * Go to open new folder, open downloaded file that you download from my repository
         * Make sure downloaded dataset, E_comerce_txt_classification.py file and review_handler all are in same folder
         * ![#1589F0](https://placehold.co/15x15/1589F0/1589F0.png) **ATTENTION!!! : root_path = "Please change the path according to your folder path" DON'T FOLLOW MY PATH IN E_comerce_txt_classification.py file SINCE THE PATH IS MY OWN FOLDER PATH**
         * Then you can run E_comerce_txt_classification.py python file

### 3. Results

1. This is LSTM model architecture with embedded dimension = 64


      ![model_architecture](https://github.com/dalila28/E-Commerce_Text_Classification/blob/main/images/model_architecture.png)



                                  Model Architecture
                                               
                                               
                                               

2. Below are the snapshot of the model performance under 100 epochs, which achiev accuracy about 99%


   
                                              

      ![model_performance1](https://github.com/dalila28/E-Commerce_Text_Classification/blob/main/images/mode_performance1.png)
      ![model_performance2](https://github.com/dalila28/E-Commerce_Text_Classification/blob/main/images/model_performance2.png)


                                                      Model Performance





3. Tensorboard snapshot showing graph of accuracy for train & test , loss for train & test





      ![tensorboard](https://github.com/dalila28/E-Commerce_Text_Classification/blob/main/images/tensorboard.png)



                                                      Tensorboard
                                             
                                             
                                             


4. Figure below showing the matplotlib graph comparison between train and test accuracy.






      ![train_vs_test](https://github.com/dalila28/E-Commerce_Text_Classification/blob/main/images/train_vs_test.png)
      
      
      
                                                  Train & Test Accuracy

### 4. Credits
1. Kaggle. (n.d.). Ecommerce Text Classification. Retrieved June, 2023, from https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification
2. For creating tensorboard, I refer tutorial from https://www.tensorflow.org/tensorboard/get_started
3. Regarding the tensorflow API that I used in my project, I always refer to this documentation https://www.tensorflow.org/api_docs/python/tf/all_symbols
