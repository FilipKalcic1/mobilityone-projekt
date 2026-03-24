"""
Performance and Scalability Analysis for MobilityOne Bot

This script analyzes current bottlenecks and provides recommendations
for handling 1000+ concurrent users.
"""

# ---
# CURRENT ARCHITECTURE ANALYSIS
# ---

CURRENT_CONFIG = {
    "worker": {
        "max_concurrent": 5,  # Limited by Azure TPM
        "memory_limit": "1GB",
        "cpu_limit": "1.0 cores",
        "redis_block_ms": 1000,
    },
    "api": {
        "memory_limit": "2GB", 
        "cpu_limit": "2.0 cores",
    },
    "redis": {
        "max_memory": "256MB",
        "eviction_policy": "allkeys-lru",
    }
}

# ---
# BOTTLENECK ANALYSIS
# ---

BOTTLENECKS = """
1. AZURE OPENAI TPM (Tokens Per Minute)
   - Current limit: ~60,000 TPM (depends on tier)
   - Average request: ~2000 tokens (input + output)
   - Max throughput: 30 requests/minute = 0.5 req/sec
   - For 1000 users: INSUFFICIENT if all active simultaneously
   
   SOLUTION: 
   - Increase Azure OpenAI quota (contact Microsoft)
   - Use multiple Azure OpenAI endpoints (load balance)
   - Implement request batching for similar queries
   - Cache frequent responses

2. SEMANTIC SEARCH (Embedding API)
   - Each routing call = 1 embedding API call
   - Additional ~500ms latency per request
   
   SOLUTION:
   - Pre-compute and cache common query embeddings
   - Use local embedding model (e.g., sentence-transformers)
   - Batch embedding requests

3. REDIS SINGLE INSTANCE
   - Current: Single Redis with 256MB
   - Risk: Single point of failure, memory pressure
   
   SOLUTION:
   - Increase Redis memory to 1GB+
   - Use Redis Cluster for HA
   - Implement Redis Sentinel

4. WORKER CONCURRENCY
   - Current: MAX_CONCURRENT = 5
   - Reason: Azure TPM limit
   
   SOLUTION:
   - Scale horizontally (multiple worker containers)
   - Use Kubernetes with KEDA autoscaler (already have k8s/keda-autoscaler.yaml)

5. DATABASE CONNECTIONS
   - Not currently pooled optimally
   
   SOLUTION:
   - Use connection pooling (PgBouncer)
   - Increase max_connections in PostgreSQL
"""

# ---
# SCALABILITY RECOMMENDATIONS
# ---

RECOMMENDATIONS = """
## For 1000+ Users:

### Tier 1: Quick Wins (No Code Changes)

1. **Increase Azure OpenAI quota**
   - Contact Microsoft for higher TPM
   - Target: 300,000 TPM = 150 req/min

2. **Scale Redis memory**
   - Change: --maxmemory 1gb
   - Add: Redis Sentinel for HA

3. **Add worker replicas**
   ```yaml
   # docker-compose.override.yml
   worker:
     deploy:
       replicas: 3
   ```

### Tier 2: Architecture Changes

1. **Use KEDA autoscaler** (k8s deployment)
   - Auto-scale workers based on Redis queue length
   - File: k8s/keda-autoscaler.yaml already exists!

2. **Local embedding model**
   - Reduce embedding API calls by 90%
   - Use: sentence-transformers/all-MiniLM-L6-v2

3. **Response caching**
   - Cache common questions (FAQ)
   - TTL: 5 minutes for data queries

### Tier 3: Full Production

1. **Multiple Azure OpenAI endpoints**
   - Load balance across 3+ endpoints
   - Different regions for redundancy

2. **Redis Cluster**
   - 3+ nodes for HA
   - Automatic failover

3. **Database read replicas**
   - Write to primary
   - Read from replicas

## Capacity Estimates:

| Users | Requests/min | Workers | Azure TPM | Redis |
|-------|-------------|---------|-----------|-------|
| 100   | 20          | 1       | 60K       | 256MB |
| 500   | 100         | 2       | 150K      | 512MB |
| 1000  | 200         | 3       | 300K      | 1GB   |
| 5000  | 1000        | 10      | 1M        | 4GB   |

## Current Status:

✅ Architecture is horizontally scalable
✅ KEDA autoscaler config exists
✅ Redis eviction policy configured
✅ Circuit breaker pattern implemented
✅ Rate limiting in place
⚠️ Single worker instance
⚠️ Azure TPM may be limiting factor
"""

# ---
# PERFORMANCE METRICS
# ---

def calculate_capacity():
    """Calculate theoretical capacity."""
    
    # Azure limits (estimate for S0 tier)
    azure_tpm = 60_000
    avg_tokens_per_request = 2000
    requests_per_minute = azure_tpm / avg_tokens_per_request
    
    # Worker limits
    max_concurrent = 5
    avg_latency_sec = 3  # 3 seconds average
    requests_per_worker_per_min = (60 / avg_latency_sec) * max_concurrent
    
    # Queue throughput
    redis_ops_per_sec = 100_000  # Redis is not the bottleneck
    
    print("=" * 60)
    print("CAPACITY ANALYSIS")
    print("=" * 60)
    print(f"Azure TPM limit: {azure_tpm:,}")
    print(f"Avg tokens/request: {avg_tokens_per_request:,}")
    print(f"Max requests/min (Azure): {requests_per_minute:.0f}")
    print(f"Max requests/min (Worker): {requests_per_worker_per_min:.0f}")
    print(f"Effective limit: {min(requests_per_minute, requests_per_worker_per_min):.0f} req/min")
    print()
    
    # User estimates (assuming 1 message per user per 5 minutes)
    messages_per_user_per_hour = 12
    effective_rpm = min(requests_per_minute, requests_per_worker_per_min)
    max_concurrent_users = (effective_rpm * 60) / messages_per_user_per_hour
    
    print(f"Assuming {messages_per_user_per_hour} messages/user/hour:")
    print(f"Max concurrent users: {max_concurrent_users:.0f}")
    print()
    
    print("TO REACH 1000 USERS:")
    needed_rpm = 1000 * messages_per_user_per_hour / 60
    print(f"  Needed: {needed_rpm:.0f} req/min")
    print(f"  Current: {effective_rpm:.0f} req/min")
    
    if needed_rpm > effective_rpm:
        workers_needed = needed_rpm / requests_per_worker_per_min
        tpm_needed = needed_rpm * avg_tokens_per_request
        print(f"  Workers needed: {workers_needed:.1f}")
        print(f"  Azure TPM needed: {tpm_needed:,.0f}")
    else:
        print(f"  ✅ Current capacity is sufficient!")


if __name__ == "__main__":
    calculate_capacity()
    print()
    print(RECOMMENDATIONS)
