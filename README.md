#  End-to-End MLOps Pipeline using Decision Tree

## Project Overview

This project demonstrates a complete **end-to-end MLOps pipeline**, integrating machine learning development with DevOps practices.  

The pipeline includes:
- Model training
- Model evaluation
- CI/CD automation
- Web application deployment
- Containerization
- Orchestration using Kubernetes

The objective is to build a **production-ready ML system** that is scalable, automated, and robust.


## Objectives

- Train a Decision Tree model using the Olivetti Faces dataset
- Automate training and testing using CI/CD (GitHub Actions)
- Deploy the model through a Flask web application
- Containerize the application using Docker
- Deploy and scale using Kubernetes (3 replicas)

## Project Structure

mlops-major/
│
├── train.py
├── test.py
├── app.py
├── requirements.txt
├── Dockerfile
├── deployment.yaml
├── service.yaml
│
├── templates/
│   └── index.html
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── README.md
├── .gitignore


## Branch Strategy

| Branch | Purpose |
|--------|--------|
| main | Initial repository setup |
| dev | ML model training + CI/CD pipeline |
| docker_cicd | Deployment using Flask, Docker, Kubernetes |

Branches are intentionally **not merged** as per assignment instructions.



##  Model Development

- Dataset: **Olivetti Faces (sklearn)**
- Algorithm: **DecisionTreeClassifier**
- Train-Test Split: **70% / 30%**
- Stratified sampling used for balanced classes
- Model saved using `joblib`



##  Model Performance

- Training Accuracy: ~ High  
- Testing Accuracy: ~85% (approx)

Slight overfitting observed (expected for Decision Tree)

---

## CI/CD Pipeline (GitHub Actions)

Workflow triggers on push to `dev` branch:

### Steps:
1. Checkout repository  
2. Setup Python environment  
3. Install dependencies  
4. Train model (`train.py`)  
5. Test model (`test.py`)  
6. Upload model artifact  
7. Upload metrics file  

👉 Ensures **automation and reproducibility**


## Flask Application

A web interface to:
- Upload face images
- Predict class using trained model
- Display prediction result

###  Features:
- Input validation
- Error handling
- Logging
- JSON-based responses
- Health check endpoint `/health`


##  Docker Implementation

The application is containerized using Docker:

###  Features:
- Lightweight base image (`python:3.9-slim`)
- Dependency management via `requirements.txt`
- Environment variable support
- Fully portable deployment



## Kubernetes Deployment

Application deployed using Kubernetes with:

###  Features:
- 3 replicas for high availability
- NodePort service for external access
- Resource limits for container efficiency
- Auto-recovery of failed pods


##  System Architecture


Developer → GitHub (dev branch)
↓
GitHub Actions (CI/CD)
↓
Model Training & Testing
↓
Docker Image
↓
Docker Hub
↓
Kubernetes Cluster
↓
Flask Web App
↓
User Upload → Prediction


## Challenges Faced

- Setting up CI/CD pipeline correctly  
- Managing Docker image builds  
- Configuring Kubernetes deployment  


## Improvements & Future Work

- Replace Decision Tree with CNN for better image performance
- Add model versioning
- Implement monitoring and logging dashboards
- Enable continuous deployment (CD)


## Conclusion

This project successfully demonstrates a **complete MLOps lifecycle**, integrating machine learning with DevOps practices.  

The system is:
- Automated  
- Scalable 
- Reproducible 
- Production-ready 