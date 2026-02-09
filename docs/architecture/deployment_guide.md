# Deployment Guide

## 1. Prerequisites

### 1.1 Infrastructure Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 16+ cores |
| **Memory** | 16GB | 64GB+ |
| **Storage** | 100GB SSD | 1TB+ SSD |
| **Network** | 1Gbps | 10Gbps |
| **Kubernetes** | 1.24+ | 1.28+ |

### 1.2 Software Dependencies
```bash
# Required Tools
- Docker 20.10+
- Kubernetes 1.24+
- kubectl 1.24+
- Helm 3.8+
- Rye (Python package manager)
- PostgreSQL 15+ with pgvector
- Redis 7.1+
- Neo4j 6.1+
```

## 2. Local Development Setup

### 2.1 Quick Start
```bash
# 1. Clone repository
git clone https://github.com/your-org/advance-rag.git
cd advance-rag

# 2. Install Rye (if not installed)
curl -sSf https://rye.astral.sh/get | bash

# 3. Install dependencies
rye sync

# 4. Setup environment
cp .env.example .env
# Edit .env with your configuration

# 5. Start local services
docker-compose up -d

# 6. Initialize databases
rye run python scripts/init_postgres.py
rye run python scripts/init_neo4j.py

# 7. Generate dummy data
rye run advance-rag generate-data --study-id STUDY001 --n-subjects 100

# 8. Start API server
rye run uvicorn advance_rag.api.main:app --reload
```

### 2.2 Docker Compose Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: advance_rag
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7.1-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  neo4j:
    image: neo4j:6.1-community
    environment:
      NEO4J_AUTH: neo4j/password123
      NEO4J_PLUGINS: '["graph-data-science"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/advance_rag
      - REDIS_URL=redis://redis:6379
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password123
    depends_on:
      - postgres
      - redis
      - neo4j
    volumes:
      - .:/app
    command: uvicorn advance_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/advance_rag
      - REDIS_URL=redis://redis:6379
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password123
    depends_on:
      - postgres
      - redis
      - neo4j
    volumes:
      - .:/app
    command: celery -A advance_rag.core.celery worker --loglevel=info

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
```

## 3. Production Deployment

### 3.1 Kubernetes Deployment

#### 3.1.1 Namespace & ConfigMaps
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: advance-rag
  labels:
    name: advance-rag

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: advance-rag-config
  namespace: advance-rag
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  DATABASE_URL: "postgresql+asyncpg://postgres:password@postgres:5432/advance_rag"
  REDIS_URL: "redis://redis:6379"
  NEO4J_URI: "bolt://neo4j:7687"
  NEO4J_USER: "neo4j"
  VECTOR_INDEX_TYPE: "hnsw"
  VECTOR_HNSW_M: "16"
  VECTOR_HNSW_EF_CONSTRUCTION: "64"
  VECTOR_SIMILARITY_THRESHOLD: "0.7"
  LLM_PROVIDER: "anthropic"
  LLM_MODEL: "claude-3-opus-20240229"
  LLM_MAX_TOKENS: "4096"
  LLM_TEMPERATURE: "0.1"
  MAX_CONCURRENT_REQUESTS: "1000"
  RATE_LIMIT_PER_MINUTE: "60"
  CACHE_TTL: "300"
  METRICS_ENABLED: "true"
  TRACING_ENABLED: "true"
```

#### 3.1.2 Secrets Management
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: advance-rag-secrets
  namespace: advance-rag
type: Opaque
data:
  DATABASE_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  NEO4J_PASSWORD: <base64-encoded-password>
  SECRET_KEY: <base64-encoded-secret>
  OPENAI_API_KEY: <base64-encoded-api-key>
  ANTHROPIC_API_KEY: <base64-encoded-api-key>
  JWT_SECRET: <base64-encoded-jwt-secret>
```

#### 3.1.3 API Deployment
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advance-rag-api
  namespace: advance-rag
  labels:
    app: advance-rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: advance-rag-api
  template:
    metadata:
      labels:
        app: advance-rag-api
    spec:
      containers:
      - name: api
        image: advance-rag:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: advance-rag-config
        - secretRef:
            name: advance-rag-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: advance-rag-api-service
  namespace: advance-rag
spec:
  selector:
    app: advance-rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### 3.1.4 Worker Deployment
```yaml
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advance-rag-worker
  namespace: advance-rag
  labels:
    app: advance-rag-worker
spec:
  replicas: 4
  selector:
    matchLabels:
      app: advance-rag-worker
  template:
    metadata:
      labels:
        app: advance-rag-worker
    spec:
      containers:
      - name: worker
        image: advance-rag:latest
        command: ["celery", "-A", "advance_rag.core.celery", "worker", "--loglevel=info", "--concurrency=4"]
        envFrom:
        - configMapRef:
            name: advance-rag-config
        - secretRef:
            name: advance-rag-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

#### 3.1.5 Database Deployments
```yaml
# k8s/postgres-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: advance-rag
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: advance_rag
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: advance-rag-secrets
              key: DATABASE_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: advance-rag
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
```

#### 3.1.6 Ingress Configuration
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: advance-rag-ingress
  namespace: advance-rag
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.advancerag.com
    secretName: advance-rag-tls
  rules:
  - host: api.advancerag.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: advance-rag-api-service
            port:
              number: 80
```

### 3.2 Helm Chart Deployment

#### 3.2.1 Chart Structure
```
helm/advance-rag/
├── Chart.yaml
├── values.yaml
├── values-prod.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── hpa.yaml
│   └── serviceaccount.yaml
└── charts/
    ├── postgresql/
    ├── redis/
    └── neo4j/
```

#### 3.2.2 Helm Values
```yaml
# helm/advance-rag/values.yaml
replicaCount: 3

image:
  repository: advance-rag
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.advancerag.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: advance-rag-tls
      hosts:
        - api.advancerag.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-password"
    database: "advance_rag"
  primary:
    persistence:
      enabled: true
      size: 100Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "secure-password"
  master:
    persistence:
      enabled: true
      size: 10Gi

neo4j:
  enabled: true
  auth:
    password: "secure-password"
  plugins:
    - "graph-data-science"
  persistence:
    enabled: true
    size: 50Gi
```

### 3.3 Deployment Commands

#### 3.3.1 Deploy with Helm
```bash
# 1. Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# 2. Deploy the application
helm install advance-rag ./helm/advance-rag \
  --namespace advance-rag \
  --create-namespace \
  --values ./helm/advance-rag/values-prod.yaml

# 3. Check deployment status
kubectl get pods -n advance-rag
kubectl get services -n advance-rag
kubectl get ingress -n advance-rag

# 4. Initialize databases
kubectl exec -it deployment/advance-rag-api -n advance-rag -- \
  python scripts/init_postgres.py

kubectl exec -it deployment/advance-rag-api -n advance-rag -- \
  python scripts/init_neo4j.py
```

#### 3.3.2 Upgrade & Rollback
```bash
# Upgrade deployment
helm upgrade advance-rag ./helm/advance-rag \
  --namespace advance-rag \
  --values ./helm/advance-rag/values-prod.yaml

# Rollback to previous version
helm rollback advance-rag 1 -n advance-rag

# Check history
helm history advance-rag -n advance-rag
```

## 4. Monitoring & Observability

### 4.1 Prometheus Monitoring
```yaml
# k8s/monitoring.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: advance-rag-metrics
  namespace: advance-rag
  labels:
    app: advance-rag-api
spec:
  selector:
    matchLabels:
      app: advance-rag-api
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s

---
apiVersion: v1
kind: Service
metadata:
  name: advance-rag-metrics
  namespace: advance-rag
  labels:
    app: advance-rag-api
spec:
  selector:
    app: advance-rag-api
  ports:
  - name: metrics
    port: 9090
    targetPort: 9090
```

### 4.2 Logging Configuration
```yaml
# k8s/logging.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: advance-rag
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*advance-rag*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name advance-rag-logs
      type_name _doc
    </match>
```

## 5. Security Configuration

### 5.1 Network Policies
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: advance-rag-network-policy
  namespace: advance-rag
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: advance-rag-worker
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: neo4j
    ports:
    - protocol: TCP
      port: 7687
```

### 5.2 Pod Security Policies
```yaml
# k8s/pod-security.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: advance-rag-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## 6. Backup & Disaster Recovery

### 6.1 Automated Backups
```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: advance-rag-backup
  namespace: advance-rag
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: advance-rag:latest
            command:
            - python
            - -c
            - |
              import asyncio
              from advance_rag.services.backup_service import backup_service
              asyncio.run(backup_service.create_full_backup())
            envFrom:
            - configMapRef:
                name: advance-rag-config
            - secretRef:
                name: advance-rag-secrets
            volumeMounts:
            - name: backup-storage
              mountPath: /backups
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### 6.2 Restore Procedure
```bash
# 1. Create restore job
kubectl apply -f k8s/restore-job.yaml

# 2. Monitor restore progress
kubectl logs -f job/advance-rag-restore -n advance-rag

# 3. Verify data integrity
kubectl exec -it deployment/advance-rag-api -n advance-rag -- \
  python -c "
import asyncio
from advance_rag.services.backup_service import backup_service
print(asyncio.run(backup_service.verify_backup('backup-2023-12-01')))
"
```

## 7. Performance Optimization

### 7.1 Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: advance-rag-api-hpa
  namespace: advance-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: advance-rag-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### 7.2 Resource Limits & Requests
```yaml
# Production resource configuration
resources:
  api:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  worker:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 4000m
      memory: 8Gi
  postgres:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 8000m
      memory: 16Gi
  redis:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  neo4j:
    requests:
      cpu: 1000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 16Gi
```

## 8. Troubleshooting

### 8.1 Common Issues & Solutions

#### 8.1.1 Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it deployment/advance-rag-api -n advance-rag -- \
  python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql+asyncpg://postgres:password@postgres:5432/advance_rag')
    print('Database connection successful')
    await conn.close()
asyncio.run(test())
"
```

#### 8.1.2 Memory Issues
```bash
# Check memory usage
kubectl top pods -n advance-rag

# Check OOM events
kubectl describe pod <pod-name> -n advance-rag | grep -i oom

# Increase memory limits
kubectl patch deployment advance-rag-api -n advance-rag -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "limits": {
              "memory": "2Gi"
            }
          }
        }]
      }
    }
  }
}'
```

#### 8.1.3 Performance Issues
```bash
# Check response times
kubectl exec -it deployment/advance-rag-api -n advance-rag -- \
  curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"

# Check database query performance
kubectl exec -it deployment/postgres -n advance-rag -- \
  psql -U postgres -d advance_rag -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

This deployment guide provides comprehensive instructions for deploying Advance RAG in both development and production environments with Kubernetes.
