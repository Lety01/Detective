
# 🧠 Detective
This project is a deep learning-based digit classifier that uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The project is built using TensorFlow and Keras, demonstrating the capability of CNNs in image recognition tasks.

## 🚀 Features

- **Data Preprocessing**: 
Normalizes pixel values to [0, 1] range.
Reshapes images to fit the CNN input requirements.
  
- **Convolutional Neural Network**:
Built with TensorFlow and Keras.
Consists of convolutional, pooling, and fully connected layers.
  
- **Model Training**:
Trained on 60,000 images of handwritten digits.
Achieves high accuracy on the test set.
  
- **Prediction and Visualization**:
Predicts digits from unseen data.
Visualizes the input image, predicted digit, and true label.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier` 

2.  **Create and activate a virtual environment**:
    
 
    
    `python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate` 
    
3.  **Install the dependencies**:
      
    `pip install -r requirements.txt` 
    

## 📊 Usage

1.  **Train the model**:

    `python src/mnist_classifier.py` 
    
2.  **Make predictions**:

    
    `python src/predict.py` 
    
3.  **Example output**: The script will display an image of a digit along with the predicted label and the true label:
   
   ![image](https://github.com/user-attachments/assets/83d62bb7-34a7-4aec-abf9-2046e6a23858)

    

## 📚 Learnings

-   **Deep Learning with CNNs**: Gained practical experience in building and training Convolutional Neural Networks.
-   **Data Preprocessing**: Learned the importance of normalizing data and preparing it for model input.
-   **Model Evaluation**: Understood how to evaluate the performance of a model using test datasets and visualize the results.

## 🤝 Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

## 📧 Contact

For any questions or suggestions, feel free to contact me at letiziagrasso01@gmail.com.


