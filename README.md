
# Self_driving_car_behavioral_cloning

This is a university project aimed at making the car in Udacity nanodegree simulator drive the left track from the simulator.
The whole specification of the assignment for this project is in the "ML D2 2023.pdf" file (3rd exercise).
To run this project you need the Udacity nanodegree simulator made in Unity. Then you need to run the model from the good_models folder. When in the root folder of the cloned project run the command 

``` python drive.py ``` 

Its the same model as model-016.h5. The other models in the folder crash but were getting better and better so I was saving them in the training process.

![driving](https://github.com/Mixa26/Self_driving_car_behavioral_cloning/blob/7a374a5bd2ba071ef49ea0b6d70e31a34590ac14/pictures/Desktop%202023.06.16%20-%2017.56.42.02_1.gif)

For this project we already had all the preprocessing steps provided to us and only had to implement the "build_model" and "train_model" in the "model.py" file. The "utils.py" has all the help functions and the "batch_generator" which is used for loading all the images and steers into code.

This is the main idea behind the training process. We drive the car around the track, record it as data where we receive images from the left, right and center of the car which we will use as features, and we also record the angle of the tires which will be our prediction target.
