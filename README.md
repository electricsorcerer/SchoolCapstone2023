Look, this is the first project I uploaded to github, and even this is only here cause it was required.
It's an image recognition program for planes, and that's kind of it.
The EDA scripts vary between data management, to chart generation, to some of the worst solutions to problems I have ever written.
The model is the trained h5 file. 
The images used to train the model aren't included, solely because they're too large altogether.
GUIproper.py is the program to run to test all this out, modelTraining.py won't run without the images present, but you can still read it, cause it's got a list of image classes used, at least the ones in the final version.
Also, for future me, the best metrics for this are "metrics=[keras.metrics.F1Score(threshold=0.5),keras.metrics.AUC()]".
AutomatedDataAugmentation was to change the dataset every training session, to reduce overfitting. 
When I finished with this project, the model was usually correct.
