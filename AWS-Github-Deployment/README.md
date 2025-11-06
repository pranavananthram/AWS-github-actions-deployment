URL for fastapi app: http://3.145.84.64/docs

For this assignment, we had to deploy our model using FastAPI, Docker, AWS and github. We first created the fastAPI app and tested it to run locally, 
then we converted it into a docker image so that we could upload it using AWS.
AWS deployment: I used cloudshell to setup the correct environment and set up parameters such as region(us-east-2) and my user ID, once that was created, I 
uploaded my files onto the application and ran it on cloudshell and created a docker image which I stored in a repo named my-fastapi-app. Once that was
created, I used ECS to setup the correct cluster (mlops-cluster) and then i created a task definition (mlops-task2) with the correct VPC(project) and once I 
completed these steps I was able to deploy it.
Issues for AWS deployment: I could not follow the same steps as shown in lecture as I was not using pycharm and did not have the AWS CLI on my computer. I had to
upload the files to cloudshell and create a repo and faced an issue where my code did not run without the excel file, this is most likely caused because, 
the docker image that I upload contains the trained model and thus I don't need the excel file if the image did not need to be created, however, since I used
cloudshell to create the docker image I also needed to upload the excel file. 
Secondly, once I had fixed the issue with the docker image being created, I had to create a new repo (my-fastapi-app) and upload my code manually into it, as
I was not able to debug the list of issues I had recieved with my previous task definition, however, upon created a new task def I did not face any previous
errors. 

Github actions deployment: Once it had been deployed on AWS, I used github actions to create a new workflow using the Deploy to Amazon ECS workflow, where I had 
to add parameters from my AWS account such as region, cluster, task definition json etc. I also had to create a github access key and secret access key using the
secret command in github repo which contains the access key to run my deployment on AWS. Once these changes have been made, any commits in my repo will cause
it to deploy on AWS to reflect these changes.
Issues for Github deployment: I faced an error in my downloaded JSON file with enableFaultInjection parameter in task definitions section and deleted it as
we are trying to run the docker image and do not need it for ECS deployment. However, upon removing it, the deployment kept referencing a past version of the json 
file and registered the error despite there not being any such parameter. Upon committing these changes and AWS updating the task definition and creating a new
version : mlops_task2:2 (second deployment) with the updated json file, this error fixed itself, however, on github deployment this was still marked as a failed
deployment until AWS task reached a stable state.
Additionally, one minor issue was that my docker file was located in a different directory from the workflow, so in the run: command I added a cd A1 command to 
change directory so that it does not mark this as an error and has access to the correct docker file. 

Please let me know if you can access the URL in case I need to reactivate it. 
