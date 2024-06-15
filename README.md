# Implementing-Few-Shot-Learning-with-CIFAR-100-Dataset
Python code to demonstrate few-shot learning on the CIFAR-100 dataset either by fine tuning a pre-trained model or a training custom model created from scratch.

Imports: Necessary libraries are imported, including PyTorch for deep learning operations, torchvision for datasets and transformations, and modules for neural network components (nn), optimizers (optim), and data handling (DataLoader).

Transforms: The transform object is defined using torchvision.transforms.Compose to apply normalization and convert images to tensors.

Dataset Loading: trainset and testset are loaded using torchvision.datasets.CIFAR100, specifying train=True for the training set and train=False for the test set. The datasets are downloaded if not already available and transformed using transform.

Model Definition: CustomModel is defined as a subclass of nn.Module. It consists of two convolutional layers (conv1 and conv2), each followed by a max pooling layer (pool). The fully connected layers (fc1 and fc2) are defined for classification into 100 classes (CIFAR-100). The forward method defines the forward pass of the model.

Training: The model is trained on the CIFAR-100 training set (trainset) using torch.utils.data.DataLoader to handle batching and shuffling of data. train_model function iterates over num_epochs epochs, computing gradients, updating weights using the SGD optimizer, and printing training loss.

Evaluation: The evaluate_model function evaluates the trained model on the CIFAR-100 test set (testset). It calculates accuracy, precision, recall, and F1 score using sklearn.metrics functions, comparing predicted and true labels.
