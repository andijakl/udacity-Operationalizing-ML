# Operationalizing an AWS ML Project

## Initial Setup

To set up the project, create a SageMaker instance. You can either choose to run a Sagemaker Studio. Personally, I prefer the UI of the SageMaker Studio instance. The setup should look like the following:

![Sagemaker Studio Instance](screenshots/sagemaker-instance.png)

To run the notebook, I chose the ml.t3.medium, as it's fast enough to run the notebook that will trigger tasks. Also, it is a fast-start enabled instance type, resulting in a shorter wait time.

### Data

Similar to the third project, I downloaded the sample data to the SageMaker Studio storage. The code in the notebook then unzips the data. Finally, I uploaded the data to a newly created s3 bucket for this project. The following screenshot shows the data uploaded to s3:

![ML data uploaded to S3 bucket](screenshots/s3.png)

## Training and Deployment (single instance)

For hyperparameter optimization, I upgraded the default framework to use to a newer version of Python (3.8) and the framework to version 1.9. Also, as the instance type of the starter project isn't available by default in my private AWS account ("ml.g4dn.xlarge", the AWS support would need to be contacted to enable this instance type, as it is rather expensive), I used the "ml.m5.2xlarge" instance type instead. It is fast enough for this training type and costs less. I left max_jobs at 2 and max_parallel_jobs at 1. 

The best hyperparameters found were: 

Next, I performed training the model with the best hyperparameters. Again, I used the "ml.m5.2xlarge" instance type.

For the predictor, I used the "ml.t2.medium", as it's cheaper than the one suggested in the sample project and was more than fast enough in the previous project to run inference on a data sample (only took a few miliseconds). 

