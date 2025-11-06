# FastAPI Model Deployment

**Live App URL:** [http://3.145.84.64/docs](http://3.145.84.64/docs)

This project demonstrates how to deploy a machine learning model using **FastAPI**, **Docker**, **AWS ECS**, and **GitHub Actions**. The deployment process involved local testing, containerization, cloud deployment, and CI/CD integration.

---

## Project Overview

- Built a FastAPI app to serve a trained ML model.
- Containerized the app using Docker.
- Deployed the container to AWS ECS using CloudShell.
- Automated future deployments via GitHub Actions.

---

## Step-by-Step Deployment

### 1. FastAPI App Development

- Created and tested the FastAPI app locally.
- Verified endpoints and model predictions using the interactive Swagger UI.

### 2. Docker Containerization

- Converted the FastAPI app into a Docker image.
- Stored the image in a repository named `my-fastapi-app`.

### 3. AWS Deployment via CloudShell

- Used **AWS CloudShell** to configure the environment:
  - Set region to `us-east-2`
  - Configured user credentials
- Uploaded project files and created the Docker image inside CloudShell.
- Pushed the image to AWS ECR.
- Set up **ECS**:
  - Created a cluster: `mlops-cluster`
  - Defined a task: `mlops-task2`
  - Assigned the correct VPC and networking settings
- Successfully deployed the containerized app.

#### AWS Deployment Challenges

- Could not follow lecture steps due to lack of AWS CLI and PyCharm.
- Had to manually upload files to CloudShell.
- Faced issues with missing Excel file during image creation:
  - The trained model was already embedded in the Docker image.
  - However, since the image was built in CloudShell, the Excel file was required.
- Resolved by uploading the Excel file and rebuilding the image.
- Created a new repository and task definition to bypass previous errors.

---

## GitHub Actions Deployment

- Set up a GitHub Actions workflow using **Deploy to Amazon ECS** template.
- Added required secrets:
  - AWS region
  - ECS cluster name
  - Task definition JSON
  - GitHub access and secret keys
- Enabled automatic deployment on every commit.

#### GitHub Deployment Challenges

- Encountered an error with `enableFaultInjection` in the task definition JSON.
  - Removed the parameter, but ECS kept referencing the old version.
  - Fixed by committing changes and allowing ECS to update to `mlops_task2:2`.
- GitHub marked deployment as failed until ECS reached a stable state.
- Dockerfile was in a subdirectory (`A1`), so added `cd A1` in the workflow to locate it correctly.

---

## Notes

- If the FastAPI app URL becomes inactive, please let me know so I can reactivate it.
- Swagger UI is available at `/docs` for testing endpoints.

---
