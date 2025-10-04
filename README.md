# DS-340-Proj

This project performs sentiment analysis on hotel reviews using deep learning models. The primary goal is to classify hotel reviews as either positive or negative based on the review text.

## Dataset

The project uses the "Datafiniti Hotel Reviews" dataset, which is expected to be in a CSV file named `Datafiniti_Hotel_Reviews.csv`. The dataset contains various information about hotel reviews, but this project primarily focuses on the `reviews.rating` and `reviews.text` columns.

## Models

The following models are implemented in this project:

* **Simple LSTM:** A basic Long Short-Term Memory network for sequence classification.
* **Stacked LSTM:** An LSTM model with two stacked LSTM layers for potentially capturing more complex patterns in the sequence data.
* **GRU:** A Gated Recurrent Unit model, which is a variation of the LSTM.
* **BERT:** The project also includes an initial exploration of using a pre-trained BERT model for sentiment analysis, although it is not fully implemented.

## Hyperparameter Tuning

Hyperparameter tuning is performed for the simple LSTM model to find the optimal combination of the following hyperparameters:
* Embedding Dimension
* LSTM Units
* Dropout Rate
* Optimizer

The tuning process uses TensorFlow's HParams API and can be monitored using TensorBoard.

## Usage

To use this project, you will need to have Python and the necessary libraries installed. The primary libraries used are pandas, scikit-learn, and TensorFlow.

1.  **Data Preparation:** Ensure the `Datafiniti_Hotel_Reviews.csv` file is in the same directory as the notebooks.
2.  **Run the Notebooks:**
    * `DS-340-Proj.ipynb`: Contains initial data exploration and an incomplete BERT implementation.
    * `340_Proj_LSTM.ipynb`: Contains the main workflow for training and evaluating the LSTM and GRU models.

## Results

The models are trained to classify hotel reviews as positive (rating >= 4) or negative (rating < 3). The performance of the models is evaluated based on their accuracy on a validation set. The `340_Proj_LSTM.ipynb` notebook saves the best performing models, which can then be used for inference on new review data.