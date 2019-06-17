# stanford_cars_dataset
This is the submission for the [Grab Computer Vision Challenge](https://www.aiforsea.com/computer-vision).

## Thought Process
It is very easy to create neural networks that can get a good accuracy on a dataset. However, most of those neural networks cannot be used in real time to perform predictions. Maybe the models are take a lot of time to perform a prediction, or maybe they are so big that they need powerful machines to run them, or maybe there is so much feature engineering and preprocessing that the model becomes very hard to scale.

It is because of this that models need to be small and fast, but still make good predictions. This was the motivation behind this submission: Can we make a model that is fast and scalable, but also robust and accurate. To do this, we used concepts of Edge Computing.

## Edge Computing
Machine Learning Models have been primarly dependent on the cloud for data storage and analysis. As the Internet of Things (IoT) and autonomous driving becomes more mainstream, the number of devices connected to the web is increasing by the millions. In fact, Forbes estimates that the number of Internet-connected devices will exceed 75 Billion by 2025. Many, if not most, of these devices, will be smart.

There are a lot of very compelling reasons for shifting computations away from the cloud and into the edge, with the most important being latency issues. Here, latency refers to the time it might take to send data to a server and then receive the response. The few seconds of delay caused by this might not be a problem for your smart home applications, but commercially, those few precious seconds, or even microseconds, can cause a machine to break down or a fatal accident.

If you have many sensors, you will probably be streaming data in the order of giga bytes every hour. Even a simple camera like a pi cam will generate nearly 50 GB of data per hour at a resolution of 1280×720 and 30 fps5. It does not make sense for companies to pay for the bandwidth to send that much data when most of it is discarded anyway. Hence, it is crucial to shift all that computation to where the data is getting generated.

In this Grab AI for SEA challenge of Computer Vision, the world of edge computing brings us closer to better model performance and buisness solution in approaching real-world problems. 

## What was done
For this work, multiple types of models were trained using transfer learning. A total of 8 models were trained: **inception_resnet, inception, mobilenet, resnet, vgg16, vgg19, xception**.

Each of these models have their own advantages and disadvantages. You can see a comparision of that in the [ipynb notebook](benchmark_vis.ipynb). The different models were trained and they were compared to see which is best based on their f1 scores and time taken to perform inference.

### Installation
The repo requires python3.6. To install the packages required, use:

```bash
pip install -r requirements.txt
```

### Training a model
The `train_model.py` file can 
