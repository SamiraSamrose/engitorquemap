# EngiTorqueMap Deployment Guide

Production deployment guide for EngiTorqueMap.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (optional, for production)
- Domain name with DNS configured
- SSL certificate

## Local Development Deployment

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/engitorquemap.git
cd engitorquemap

# Create environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 2. Start Services
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### 3. Verify Deployment
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Access dashboard
open http://localhost:8000
```

## Production Deployment (Docker)

### 1. Prepare Production Environment
```bash
# Create production directory
mkdir -p /opt/engitorquemap
cd /opt/engitorquemap

# Clone repository
git clone https://github.com/yourusername/engitorquemap.git .

# Configure environment
cp .env.example .env
nano .env
```

**Production .env**:
```bash
DEBUG=False
DATABASE_URL=postgresql://user:password@postgres:5432/engitorquemap
REDIS_PASSWORD=strong_redis_password
OPENAI_API_KEY=sk-your-production-key
```

### 2. Build Images
```bash
docker-compose -f docker/docker-compose.yml build
```

### 3. Start Production Services
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 4. Setup Nginx Reverse Proxy

**nginx.conf**:
```nginx
upstream backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name engitorquemap.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name engitorquemap.com;

    ssl_certificate /etc/ssl/certs/engitorquemap.crt;
    ssl_certificate_key /etc/ssl/private/engitorquemap.key;

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 5. Setup SSL with Let's Encrypt
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d engitorquemap.com

# Auto-renewal
sudo certbot renew --dry-run
```

## Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace engitorquemap
```

### 2. Deploy PostgreSQL

**postgres-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: engitorquemap
spec:
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
        image: timescale/timescaledb:latest-pg14
        env:
        - name: POSTGRES_DB
          value: engitorquemap
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: engitorquemap
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### 3. Deploy Backend

**backend-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: engitorquemap
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: engitorquemap/backend:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: backend-secrets
              key: database-url
        - name: REDIS_HOST
          value: redis
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: kafka:9092
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend
  namespace: engitorquemap
spec:
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### 4. Deploy Ingress

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: engitorquemap-ingress
  namespace: engitorquemap
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/websocket-services: backend
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - engitorquemap.com
    secretName: engitorquemap-tls
  rules:
  - host: engitorquemap.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
```

### 5. Apply Configurations
```bash
# Create secrets
kubectl create secret generic db-credentials \
  --from-literal=username=engitorque_user \
  --from-literal=password=secure_password \
  -n engitorquemap

# Apply deployments
kubectl apply -f postgres-deployment.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f ingress.yaml

# Check status
kubectl get pods -n engitorquemap
kubectl get services -n engitorquemap
```

## Monitoring Setup

### Prometheus
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: engitorquemap
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'backend'
      static_configs:
      - targets: ['backend:8000']
```

### Grafana Dashboard

Import pre-built dashboard:
```bash
# Access Grafana
kubectl port-forward -n engitorquemap svc/grafana 3000:3000

# Open http://localhost:3000
# Import dashboard from dashboard.json
```

## Backup Strategy

### Database Backup
```bash
# Automated daily backup
cat > /opt/engitorquemap/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/backups
docker exec postgres pg_dump -U engitorque_user engitorquemap | gzip > $BACKUP_DIR/backup_$DATE.sql.gz
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
EOF

chmod +x /opt/engitorquemap/backup.sh

# Add to crontab
crontab -e
# Add: 0 2 * * * /opt/engitorquemap/backup.sh
```

### ML Models Backup
```bash
# Backup trained models
tar -czf ml_models_backup.tar.gz ml_models/trained/
aws s3 cp ml_models_backup.tar.gz s3://engitorquemap-backups/
```

## Disaster Recovery

### Recovery Procedure
```bash
# 1. Restore database
gunzip < backup_20240115_020000.sql.gz | docker exec -i postgres psql -U engitorque_user engitorquemap

# 2. Restore ML models
aws s3 cp s3://engitorquemap-backups/ml_models_backup.tar.gz .
tar -xzf ml_models_backup.tar.gz

# 3. Restart services
docker-compose -f docker/docker-compose.yml restart
```

## Performance Tuning

### PostgreSQL Optimization
```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET random_page_cost = 1.1;

-- Restart required
```

### Redis Optimization
```bash
# redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""
```

### Application Scaling
```bash
# Scale backend pods
kubectl scale deployment backend --replicas=5 -n engitorquemap

# Auto-scaling
kubectl autoscale deployment backend \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n engitorquemap
```

## Troubleshooting

### Check Logs
```bash
# Docker
docker-compose -f docker/docker-compose.yml logs -f backend

# Kubernetes
kubectl logs -f deployment/backend -n engitorquemap
```

### Common Issues

**Database Connection Failed**:
```bash
# Check PostgreSQL status
kubectl get pods -n engitorquemap | grep postgres
kubectl logs postgres-xxx -n engitorquemap
```

**Kafka Not Connecting**:
```bash
# Check Kafka status
kubectl exec -it kafka-0 -n engitorquemap -- kafka-topics --list --bootstrap-server localhost:9092
```

**High Memory Usage**:
```bash
# Check resource usage
kubectl top pods -n engitorquemap

# Increase limits
kubectl edit deployment backend -n engitorquemap
```

## Maintenance

### Update Application
```bash
# Pull latest code
git pull origin main

# Rebuild images
docker-compose -f docker/docker-compose.yml build

# Rolling update
docker-compose -f docker/docker-compose.yml up -d
```

### Database Maintenance
```bash
# Vacuum database
docker exec postgres psql -U engitorque_user -d engitorquemap -c "VACUUM ANALYZE;"

# Reindex
docker exec postgres psql -U engitorque_user -d engitorquemap -c "REINDEX DATABASE engitorquemap;"
```

## Security Checklist

- [ ] Change default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Setup API authentication
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Network segmentation
- [ ] Access control policies
- [ ] Penetration testing