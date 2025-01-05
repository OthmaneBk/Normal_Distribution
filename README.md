To detect fraudulent or non-fraudulent activities, we use the Probability Density Function of a normal distribution to calculate the probability of each feature belonging to a particular class (fraudulent or non-fraudulent), each feature follows a normal distribution with parameters Mean and Standard deviation.

![image](https://github.com/user-attachments/assets/716cc2ff-42d7-4508-8585-6dbdbc7d03f4)

We compute the product of all the probabilities. Finaly we fix a Threshold to decide whether an observation is considered "fraudulent" or "non-fraudulent"


![image](https://github.com/user-attachments/assets/54b0a241-0917-40ca-bd2e-2ece2d930cbb)


