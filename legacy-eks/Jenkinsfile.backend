pipeline {
  agent any

  // All 3 jobs share this lock, so builds run one at a time (needs the Lockable Resources plugin)
  options {
    lock('docker-build')
  }

  environment {
    AWS_REGION   = 'us-east-1'
    ECR_REGISTRY = '389517403340.dkr.ecr.us-east-1.amazonaws.com' // AWS ACCOUNT_ID
    CLUSTER      = 'nexus-ai'
    IMAGE        = "${ECR_REGISTRY}/nexus-ai-backend:${GIT_COMMIT}"
    POSTGRES_IMAGE = "${ECR_REGISTRY}/nexus-ai-backend:postgres-pg15-vector-0.8.6-r1"
  }

  stages {
    // Run unit tests in a throwaway node container (fails the build if they fail)
    stage('Test') {
      steps {
        sh 'docker run --rm -v $WORKSPACE:/app -w /app node:24-alpine sh -c "npm ci && npm test -- --runInBand && npm run build"'
      }
    }

    stage('Build image') {
      steps {
        sh 'docker build -t $IMAGE .'
        sh '''
          if ! aws ecr describe-images --repository-name nexus-ai-backend --image-ids imageTag=postgres-pg15-vector-0.8.6-r1 >/dev/null 2>&1; then
            docker build -f Kubernetes/postgres.Dockerfile -t $POSTGRES_IMAGE .
          fi
        '''
      }
    }

    stage('Sprint 5 lifecycle integration') {
      steps {
        sh '''
          set -eu
          NETWORK="nexus-s5-${BUILD_NUMBER}"
          POSTGRES="nexus-s5-postgres-${BUILD_NUMBER}"
          REDIS="nexus-s5-redis-${BUILD_NUMBER}"
          BACKEND="nexus-s5-backend-${BUILD_NUMBER}"
          PG_IMAGE="nexus-postgres-ci:${BUILD_NUMBER}"
          cleanup() {
            docker rm -f "$BACKEND" "$REDIS" "$POSTGRES" >/dev/null 2>&1 || true
            docker network rm "$NETWORK" >/dev/null 2>&1 || true
            docker rmi "$PG_IMAGE" >/dev/null 2>&1 || true
          }
          trap cleanup EXIT
          cleanup
          docker network create "$NETWORK" >/dev/null
          docker build -f Kubernetes/postgres.Dockerfile -t "$PG_IMAGE" .
          docker run -d --name "$POSTGRES" --network "$NETWORK" \
            -e POSTGRES_USER=nexus -e POSTGRES_PASSWORD=nexus_ci -e POSTGRES_DB=nexus \
            "$PG_IMAGE" >/dev/null
          docker run -d --name "$REDIS" --network "$NETWORK" redis:7-alpine >/dev/null

          for attempt in $(seq 1 60); do
            if docker exec "$POSTGRES" pg_isready -U nexus -d nexus >/dev/null 2>&1 && \
               docker exec "$REDIS" redis-cli ping >/dev/null 2>&1; then break; fi
            if [ "$attempt" -eq 60 ]; then echo 'Postgres/Redis did not become ready'; exit 1; fi
            sleep 1
          done

          docker run --rm --network "$NETWORK" \
            -e DATABASE_URL=postgresql://nexus:nexus_ci@${POSTGRES}:5432/nexus \
            -e DATABASE_SSL=false "$IMAGE" \
            node node_modules/typeorm/cli.js -d dist/database/data-source.js migration:run
          docker run --rm --network "$NETWORK" \
            -e DATABASE_URL=postgresql://nexus:nexus_ci@${POSTGRES}:5432/nexus \
            -e DATABASE_SSL=false "$IMAGE" node dist/database/seeds/seed-admin.js
          docker run --rm --network "$NETWORK" \
            -e DATABASE_URL=postgresql://nexus:nexus_ci@${POSTGRES}:5432/nexus \
            -e DATABASE_SSL=false "$IMAGE" node dist/database/seeds/seed-demo.js

          docker run -d --name "$BACKEND" --network "$NETWORK" \
            -e NODE_ENV=test -e PORT=3000 \
            -e DATABASE_URL=postgresql://nexus:nexus_ci@${POSTGRES}:5432/nexus \
            -e DATABASE_SSL=false -e REDIS_URL=redis://${REDIS}:6379 \
            -e QUEUES_ENABLED=true -e AI_MOCK_MODE=true \
            -e EVALUATION_SANDBOX_MODE=http \
            -e JWT_SECRET=integration-only-secret-with-more-than-32-characters \
            -e FRONTEND_URL=http://frontend.integration \
            -e AI_SERVICE_URL=http://unused:8000 \
            -e GOOGLE_CLIENT_ID=integration -e GOOGLE_CLIENT_SECRET=integration \
            -e GOOGLE_CALLBACK_URL=http://backend.integration/api/auth/google/callback \
            -e CLOUDINARY_CLOUD_NAME=integration -e CLOUDINARY_API_KEY=integration \
            -e CLOUDINARY_API_SECRET=integration -e STRIPE_SECRET_KEY=sk_test_integration \
            "$IMAGE" >/dev/null

          for attempt in $(seq 1 90); do
            if docker run --rm --network "$NETWORK" node:24-alpine \
              node -e "fetch('http://${BACKEND}:3000/api/health').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"; then break; fi
            if [ "$attempt" -eq 90 ]; then docker logs "$BACKEND"; exit 1; fi
            sleep 1
          done
          docker run --rm --network "$NETWORK" \
            -e API_BASE=http://${BACKEND}:3000/api \
            -v "$WORKSPACE/test/scripts:/tests:ro" node:24-alpine \
            node /tests/sprint5-lifecycle.mjs
        '''
      }
    }

    stage('Push to ECR') {
      steps {
        sh 'aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY'
        sh 'docker push $IMAGE'
        sh '''
          if ! aws ecr describe-images --repository-name nexus-ai-backend --image-ids imageTag=postgres-pg15-vector-0.8.6-r1 >/dev/null 2>&1; then
            docker push $POSTGRES_IMAGE
          fi
        '''
      }
    }

    stage('Deploy to EKS') {
      steps {
        sh 'aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER'
        sh 'kubectl apply -f Kubernetes/evaluation-sandbox-rbac.yaml'
        sh 'kubectl apply -f Kubernetes/evaluation-sandbox-networkpolicy.yaml'
        sh 'kubectl apply -f Kubernetes/configmap.yaml'
        sh 'kubectl apply -f Kubernetes/Services/postgres-service.yaml'
        sh 'sed "s|nexus-ai-postgres:local|$POSTGRES_IMAGE|" Kubernetes/Deployments/postgres-statefulset.yaml | kubectl apply -f -'
        sh 'kubectl rollout status statefulset/postgres --timeout=5m'
        sh 'kubectl apply -f Kubernetes/Deployments/backend-deployment.yaml'
        sh 'kubectl set image deployment/backend backend=$IMAGE migrate=$IMAGE'
        sh 'kubectl rollout status deployment/backend --timeout=5m'
      }
    }
  }

  // Free disk space after every build (pass or fail)
  post {
    always {
      sh 'docker rmi $IMAGE || true'
      sh 'docker rmi $POSTGRES_IMAGE || true'
      sh 'docker system prune -f'
    }
  }
}
