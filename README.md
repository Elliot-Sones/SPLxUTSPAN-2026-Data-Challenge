# SPLxUTSPAN-2026-Data-Challenge

https://github.com/Elliot-Sones/SPLxUTSPAN-2026-Data-Challenge.git

## Competition Description: 

Basketball shooting research currently relies heavily on understanding and analyzing the shot outcome patterns of players. Are they shooting with an optimal angle? Do they have a right- or left-bias in their shots? What is their consistency in achieving the optimal depth for their shots?

While these questions are interesting, the measurement of these variables describes the end-product of the shot, what happens to the ball when it has already left the hands of the player. A shot is not just the end-product of the action, it's everything encompassing the movement of the player and the ball across the whole movement.

Over the course of 1-2 seconds, a basketball player will coordinate their joints and apply forces in various ways to propel a basketball towards the hoop. While the most basic basketball research focuses solely on if the shot was made or missed, the next level focuses on how the shot was made or missed (ball landing outcomes) — we are interested in why these shot outcomes occur.

For the 2026 SPLxUTSPAN Data Challenge, we are interested in models that use biomechanical features to predict basketball free throw shot landing outcomes. You will be provided a dataset of over 450 basketball free throws tracked using a markerless motion capture system, with shot landing outcomes provided for each shot. Your goal is to use the keypoint features and shot outcome features in the training dataset to develop a model that uses the movements of the shooter's body to predict how the ball is going to land on the hoop in three features: 1) Angle, 2) Depth, 3) Left/Right.

Please navigate to the data page for a full description of the training and test datasets.

The full dataset will be released after the conclusion of the data challenge.



## Evaluation Criteria

Submissions are quantitatively evaluated on Mean Squared Error (https://en.wikipedia.org/wiki/Mean_squared_error) between the predicted and actual SCALED shot outcome variables (Angle, Depth, Left/Right). As our outcome values have different ranges of values, we are evaluating error based on SCALED versions of the outcomes. More details on this important distinction are explained below.

Data Challenge submissions will be scored 60% from the quantitative results from your leaderboard submission, and 40% qualitatively from the methodological write-up in your submission.

## Quantitative Submission

For each shot ID in the test set, you must predict a value for each of three SCALED target variables (Angle, Depth, Left/Right). You must upload a prediction of the shot outcomes in test data to Kaggle and have it evaluated on the leaderboard. The file should contain a header and have the following format:

id,scaled_angle,scaled_depth,scaled_left_right
1,0,0,0
2,0,0,0 
3,0,0,0 
etc.

