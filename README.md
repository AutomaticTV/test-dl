# ATV Deep Learning engineer test #

### Task ###

Technical interview for AutomaticTV, Deep Learning engineer position.

The proposed coding interview is settled as a typical object detection problem using a subset of a custom dataset.
The data is organized as a frame - XML pair. To receive the data, please ask the interviewing committee.

The first part consists on building the dataset reader function from the XML files, the second part consists on 
building the training loop of the object detector. 



### How do I get set up? ###
This test is built upon python 3.6, take that into account when building the virtual environment.

    pip install -r requirements.txt

If needed, feel free to use any other packages.

Create data and models directories.

    mkdir data models

Unzip the provided file in the **data** folder.

### TODO: ###
Complete the provided code in python and pytorch. 
1. Complete the dataset reader function **parse_info_from_xml** in **dataset.py**
2. Build the models, loss function, and training loop in **train.py**. 
_Hint: if needed you can use [this](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) tutorial on object detection with pytorch_
3. Extra points: build the testing loop on the provided video.

### Data structure ###
Inside folder **data**:

    |-- match1
        |-- file1.jpg
        |-- file1.xml
        ...
        |-- fileN.jpg
        |-- fileN.xml
       
    |-- matchK
        |-- file1.jpg
        |-- file1.xml
        ...
        |-- fileN.jpg
        |-- fileN.xml
    

        
## Aspects to evaluate
### Language and libraries 
* Python
* Pytorch
* OpenCV

### Deliverables
* Source code
* Results
* Optional: a file with observed things and tensorboard results (graphs, loss functions, ...)

### What we value
* Coding structure and habits
* Solvency given a problem

### What we don't assess
* What extra packages are chosen, if any

### Recommendations:
* Using already imported packages (hint: they are hints!)
* Using already created variables
* Using documentation
