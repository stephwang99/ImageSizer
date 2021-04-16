# ImageSizer
The purpose of the project is to scale object images appropriately into a background image.

# Install and Run
1. Ensure all dependencies are resolved by using the requirements.txt file
   `pip install -r requirements.txt`
2. Clone the repository 
   `git clone https://github.com/stephwang99/ImageSizer.git`
3. Download the weights dataset from https://github.com/experiencor/raccoon_dataset
4. Create the Keras model by running `python3 yolo3.py`
   You can also download the Keras model from their github https://github.com/experiencor/keras-yolo3
5. You should notice a file `model.h5` created in your folder.
6. Run the program by providing a path to the object image 
   `python3 project.py --path <path_name>`
