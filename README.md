# EiffelTower vs WatArun (Temper of Dawn)

Eiffel Tower is a famous landmark in Paria, France. Wat Arun or the Temple of Dawn is a famous landmark along the Chaopraya River in Bangkok, Thailand. They has similar structure, but very different architecture. This project is trying to classify pictures of these two places with convolution neural network.

Images are obtain by mining google images of the two places.There are about 600 pictures, which are randomly splitted into train (70%), test(20%) and valid (10%) folders. I implement the procedure I learned from fast.ai.

Skills

•	Mining images from google images 
•	Dataloader and image augmentation in GPU (fastai, py torch)
•	Transfer learning using pretrained resnet34 model.
•	Achieve 93% accuracy 

# Picture of Wat Arun
![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig2.png)

# Picture of Eiffel Tower
![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig1.png)

I played around with the batch size (bs) and sz. The former because I don't have a large set of images and the later because a lot of Wat Arun images are not square. Then obtain 98% accuracy, after the 4th epoch. Let's explore the results. Pick random images from the validation set to see what is most confident and lease confident classifications. Note that Eiffel is labeled as 0 and Wat Arun is labeled as 1.

**Correctly classified**

![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig3.png)

**Incorrectly classified**

![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig4.png)

**Most Incorrect Wat Arun**

![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig6.png)

**Most Incorrect Eiffel**

![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig5.png)

Most incorrect results are due to blur image or very far away, making the structures very similar. Next, I improve the model by choosing a learning rate, and data agumentation. 

# Optimizing the model: data augmentation
Data augmentation randomly change the images by: flipping, zooming, rotation, stretching, changing lighting parameters, and padding. I use fastai library to play with augmentation: adding rotation, change lighting, and add padding into the training sets.
These are augmented images. 

![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig7.png)

After image augmentation, I am able to slightly improve the accuracy to 97%. Here is the confusion matrix.

![](https://github.com/worasom/EiffelTower_vs_WatArun/blob/master/gitfigures/fig8.png)

Each classes have about the same miss-classified images. Given that there are less images of Wat Arun from the internet, thus the model has more difficult time classifying Wat Arun. 

## Summary

In this document, I use a CNN and pre-train resnet34 model to diccern image of Eiffel Tower and Wat Arun. I got about 96% accuracy. Using a fast.ai, augmentation library slightly improve the accuracy, while using differential learning rate does not help. The next step is trying another models, playing with augmentation, adding more training sets.

