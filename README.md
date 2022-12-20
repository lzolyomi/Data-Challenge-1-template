# Data-Challenge-1-template
This repository contains the template code for the TU/e course JBG040 Data Challenge 1

## Code structure
The sample code is structured into multiple files, based on their functionality. 
There are six `.py` files in total, each containing a different part of the code. 

- To download the data: run the `ImageDataset.py` file. The script will create a directory `/data/` and download the training and test data with corresponding labels to this directory. 
    - You will only have to run this script once usually, at the beginning of your project.

- To run the whole training/evaluation pipeline: run `main.py`. This script is prepared to:
    - Load your train and test data (Make sure its downloaded beforehand!)
    - Initializes the neural network as defined in the `Net.py` file.
    - Initialize loss functions and optimizers. If you want to change the loss function/optimizer, do it here.
    - Define number of training epochs and batch size
    - Check and enable GPU acceleration for training.
    - Train the neural network and perform evaluation on test set at the end of each epoch.
    - Finally, save your trained model's weights so that you can reload them.


## GitHub setup instructions
1. Click the green *<> Code* button at the upper right corner of the repositiory.
2. Make sure that the tab *Local* is selected and click *Download ZIP*.
3. Go to the GitHub homepage and create a new repository.
4. Make sure that the repository is set to **private** and give it the name **JBG040-GroupXX**, where XX is your group number.
5. Press *uploading an exisiting file* and upload the extracted files from Data-Challenge-1-template-main.zip to your repository. Note that for the initial commit you should commit directly to the main branch
6. Invite your **group members, tutor and teachers** by going to *Settings > Collaborators > Add people*.
7. Open PyCharm and make sure that your GitHub account is linked.*
8. In the welcome screen of PyCharm, click *Get from VCs > GitHub* and select your repository and click on clone.
9. After the repository is cloned, you can now create a virtual environment using the requirements.txt.

*For information on how to install PyCharm and link Github to your PyCharm, we refer to the additional resources page on Canvas.


## Environment setup instructions
We recommend to set up a virtual Python environment to install all necessary packages. 
These packages are included in the `requirements.txt` file.

## Submission instructions
After each sprint, you are expected to submit your code. This will **not** be done in Canvas, instead you will be creating a release of your current repository. 
**Note that you are not allowed to update your old releases after the deadline.**

1. Make sure that your code is running without issues and that **everything is pushed to the main branch**.
2. Head over to your repository and click on *Releases* (located at the right-hand side).
3. Click on the green button *Create a new release*.*
4. Click on *Choose a tag*.
5. Fill in the textbox with **SprintX** where X is the current sprint number and press *Create new tag: SprintX*.
6. Fill in the title of the release with **Group XX Sprint X** where XX is your group number and X is the current sprint number.
7. Click the green *Publish release* button.
8. **Verify** that your release has been succesfully created by heading over to your repository and press the *Releases* button once again. There you should be able to see your newly created release

*After the first release, you should click *Draft a new release* instead of *Create a new release*
