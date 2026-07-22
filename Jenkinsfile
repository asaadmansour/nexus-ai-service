pipeline {
  agent any

  environment {
    AWS_REGION   = 'us-east-1'
    ECR_REGISTRY = '389517403340.dkr.ecr.us-east-1.amazonaws.com' // AWS ACCOUNT_ID
    CLUSTER      = 'nexus-ai'
    IMAGE        = "${ECR_REGISTRY}/nexus-ai-service:${GIT_COMMIT}"
  }

  stages {
    // No unit tests in this repo yet; add a Test stage here when they exist.

    stage('Build image') {
      steps {
        sh 'docker build -t $IMAGE .'
      }
    }

    // Security gate: FAIL on fixable HIGH/CRITICAL before the image is pushed
    stage('Scan (Trivy)') {
      steps {
        sh 'docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image --exit-code 1 --severity HIGH,CRITICAL --ignore-unfixed $IMAGE'
      }
    }

    stage('Push to ECR') {
      steps {
        sh 'aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY'
        sh 'docker push $IMAGE'
      }
    }

    stage('Deploy to EKS') {
      steps {
        sh 'aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER'
        // First run: kubectl apply the Kubernetes/ manifests. After that, just swap the image tag:
        sh 'kubectl set image deployment/ai-service ai-service=$IMAGE'
      }
    }
  }

  // Free disk space after every build (pass or fail)
  post {
    always {
      sh 'docker rmi $IMAGE || true'
      sh 'docker system prune -f'
    }
  }
}
