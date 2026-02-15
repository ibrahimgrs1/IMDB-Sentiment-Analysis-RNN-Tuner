🎬 IMDB Sentiment Analysis with RNN & Keras Tuner
This repository contains a Deep Learning project focused on Sentiment Analysis of IMDB movie reviews. The project utilizes Recurrent Neural Networks (RNN) and leverages Keras Tuner for automated hyperparameter optimization to achieve the most efficient model architecture.

🚀 Key Features
Recurrent Architecture: Implements a SimpleRNN to capture sequential dependencies in text data.

Automated Tuning: Uses Keras Tuner (RandomSearch) to optimize neurons, embedding dimensions, and optimizers, moving beyond manual trial and error.

Data Preprocessing: Efficiently handles text tokenization and padding for the top 10,000 most frequent words in the IMDB dataset.

Performance Metrics: Evaluated using both Accuracy and AUC (Area Under Curve) to ensure reliable classification performance.

🏗️ Model Architecture
The neural network consists of the following layers:

Embedding Layer: Maps word indices to dense vectors of a fixed size.

SimpleRNN Layer: Processes sequences and maintains a hidden state to understand context.

Dropout Layer: Applied dynamically (ranging from 0.2 to 0.6) to prevent overfitting.

Dense (Output) Layer: Uses a sigmoid activation function for binary sentiment classification (Positive/Negative).

📊 Hyperparameter OptimizationThe Keras Tuner explored the following search space to find the "champion" model:ParameterSearch RangeEmbedding Output Dim32, 64, 96, 128RNN Units32, 64, 96, 128Dropout Rate0.2 to 0.6 (step=0.1)OptimizerAdam, RMSprop

📈 Performance Results
After the tuning process, the best model achieved:

Test Accuracy: [INSERT_ACCURACY_HERE]%

Test AUC Score: [INSERT_AUC_HERE]

Best Validation Loss: 0.4780 (Found in Trial #1)
