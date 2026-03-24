#!/bin/bash
# =============================================================================
# Create Sealed Secret for MobilityOne WhatsApp Bot
#
# Prerequisites:
#   - kubectl configured with cluster access
#   - kubeseal CLI installed (brew install kubeseal)
#   - Sealed Secrets controller deployed in cluster
#
# Usage:
#   chmod +x k8s/create-sealed-secret.sh
#   ./k8s/create-sealed-secret.sh
# =============================================================================
set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
SECRET_NAME="mobility-secrets"
OUTPUT_FILE="k8s/sealed-secrets-generated.yaml"
TEMP_FILE=$(mktemp)

# Cleanup on exit
trap 'rm -f "$TEMP_FILE"' EXIT

echo "============================================"
echo " MobilityOne Bot - Sealed Secret Generator"
echo "============================================"
echo ""
echo "Namespace: $NAMESPACE"
echo "Secret:    $SECRET_NAME"
echo ""
echo "Enter values for each secret. Press Enter to skip (use existing value)."
echo ""

# Helper function to prompt for secret value
prompt_secret() {
    local key="$1"
    local description="$2"
    local default_hint="${3:-}"
    local value

    if [ -n "$default_hint" ]; then
        echo -n "  $key ($description) [$default_hint]: "
    else
        echo -n "  $key ($description): "
    fi
    read -r -s value
    echo ""

    if [ -n "$value" ]; then
        echo "$value"
    elif [ -n "$default_hint" ] && [ "$default_hint" != "REQUIRED" ]; then
        echo "$default_hint"
    fi
}

echo "--- Database ---"
DB_URL=$(prompt_secret "DATABASE_URL" "Bot user connection string" "REQUIRED")
ADMIN_DB_URL=$(prompt_secret "ADMIN_DATABASE_URL" "Admin user connection string" "REQUIRED")

echo ""
echo "--- Redis ---"
REDIS_URL=$(prompt_secret "REDIS_URL" "Redis connection string" "REQUIRED")

echo ""
echo "--- Azure OpenAI ---"
OPENAI_KEY=$(prompt_secret "AZURE_OPENAI_API_KEY" "API key" "REQUIRED")
OPENAI_ENDPOINT=$(prompt_secret "AZURE_OPENAI_ENDPOINT" "Endpoint URL" "https://m1-ai-dev.openai.azure.com/")

echo ""
echo "--- MobilityOne API ---"
M1_CLIENT_ID=$(prompt_secret "MOBILITY_CLIENT_ID" "OAuth2 client ID" "m1AI")
M1_CLIENT_SECRET=$(prompt_secret "MOBILITY_CLIENT_SECRET" "OAuth2 client secret" "REQUIRED")
M1_TENANT_ID=$(prompt_secret "MOBILITY_TENANT_ID" "Tenant ID" "dee707eb-66ad-42e1-92f2-068be031f18a")
M1_API_URL=$(prompt_secret "MOBILITY_API_URL" "API base URL" "https://dev-k1.mobilityone.io/")
M1_AUTH_URL=$(prompt_secret "MOBILITY_AUTH_URL" "OAuth2 token URL" "https://dev-k1.mobilityone.io/sso/connect/token")

echo ""
echo "--- Infobip (WhatsApp) ---"
IB_API_KEY=$(prompt_secret "INFOBIP_API_KEY" "API key" "REQUIRED")
IB_BASE_URL=$(prompt_secret "INFOBIP_BASE_URL" "Base URL" "yr1m3g.api.infobip.com")
IB_SENDER=$(prompt_secret "INFOBIP_SENDER_NUMBER" "Sender number" "12172817448")
IB_SECRET=$(prompt_secret "INFOBIP_SECRET_KEY" "Webhook secret" "REQUIRED")
WA_VERIFY=$(prompt_secret "WHATSAPP_VERIFY_TOKEN" "Webhook verify token" "REQUIRED")

echo ""
echo "--- Admin API ---"
ADMIN_T1=$(prompt_secret "ADMIN_TOKEN_1" "Admin token 1 (64-char hex)" "REQUIRED")
ADMIN_T1_USER=$(prompt_secret "ADMIN_TOKEN_1_USER" "Username for token 1" "filip.kalcic")
ADMIN_T2=$(prompt_secret "ADMIN_TOKEN_2" "Admin token 2 (64-char hex)" "REQUIRED")
ADMIN_T2_USER=$(prompt_secret "ADMIN_TOKEN_2_USER" "Username for token 2" "damir.skrtic")

echo ""
echo "--- Monitoring ---"
SENTRY_DSN=$(prompt_secret "SENTRY_DSN" "Sentry DSN" "")

echo ""
echo "--- GDPR ---"
GDPR_SALT=$(prompt_secret "GDPR_HASH_SALT" "64-char hex salt" "REQUIRED")

echo ""
echo "Creating plain secret..."

# Build kubectl command
CMD="kubectl create secret generic $SECRET_NAME --namespace=$NAMESPACE"
[ -n "$DB_URL" ] && CMD="$CMD --from-literal=DATABASE_URL=$DB_URL"
[ -n "$ADMIN_DB_URL" ] && CMD="$CMD --from-literal=ADMIN_DATABASE_URL=$ADMIN_DB_URL"
[ -n "$REDIS_URL" ] && CMD="$CMD --from-literal=REDIS_URL=$REDIS_URL"
[ -n "$OPENAI_KEY" ] && CMD="$CMD --from-literal=AZURE_OPENAI_API_KEY=$OPENAI_KEY"
[ -n "$OPENAI_ENDPOINT" ] && CMD="$CMD --from-literal=AZURE_OPENAI_ENDPOINT=$OPENAI_ENDPOINT"
[ -n "$M1_CLIENT_ID" ] && CMD="$CMD --from-literal=MOBILITY_CLIENT_ID=$M1_CLIENT_ID"
[ -n "$M1_CLIENT_SECRET" ] && CMD="$CMD --from-literal=MOBILITY_CLIENT_SECRET=$M1_CLIENT_SECRET"
[ -n "$M1_TENANT_ID" ] && CMD="$CMD --from-literal=MOBILITY_TENANT_ID=$M1_TENANT_ID"
[ -n "$M1_API_URL" ] && CMD="$CMD --from-literal=MOBILITY_API_URL=$M1_API_URL"
[ -n "$M1_AUTH_URL" ] && CMD="$CMD --from-literal=MOBILITY_AUTH_URL=$M1_AUTH_URL"
[ -n "$IB_API_KEY" ] && CMD="$CMD --from-literal=INFOBIP_API_KEY=$IB_API_KEY"
[ -n "$IB_BASE_URL" ] && CMD="$CMD --from-literal=INFOBIP_BASE_URL=$IB_BASE_URL"
[ -n "$IB_SENDER" ] && CMD="$CMD --from-literal=INFOBIP_SENDER_NUMBER=$IB_SENDER"
[ -n "$IB_SECRET" ] && CMD="$CMD --from-literal=INFOBIP_SECRET_KEY=$IB_SECRET"
[ -n "$WA_VERIFY" ] && CMD="$CMD --from-literal=WHATSAPP_VERIFY_TOKEN=$WA_VERIFY"
CMD="$CMD --from-literal=VERIFY_WHATSAPP_SIGNATURE=true"
[ -n "$ADMIN_T1" ] && CMD="$CMD --from-literal=ADMIN_TOKEN_1=$ADMIN_T1"
[ -n "$ADMIN_T1_USER" ] && CMD="$CMD --from-literal=ADMIN_TOKEN_1_USER=$ADMIN_T1_USER"
[ -n "$ADMIN_T2" ] && CMD="$CMD --from-literal=ADMIN_TOKEN_2=$ADMIN_T2"
[ -n "$ADMIN_T2_USER" ] && CMD="$CMD --from-literal=ADMIN_TOKEN_2_USER=$ADMIN_T2_USER"
[ -n "$SENTRY_DSN" ] && CMD="$CMD --from-literal=SENTRY_DSN=$SENTRY_DSN"
[ -n "$GDPR_SALT" ] && CMD="$CMD --from-literal=GDPR_HASH_SALT=$GDPR_SALT"

# Generate plain YAML (not applied to cluster)
eval "$CMD --dry-run=client -o yaml" > "$TEMP_FILE"

echo "Sealing secret with kubeseal..."
kubeseal --format yaml < "$TEMP_FILE" > "$OUTPUT_FILE"

echo ""
echo "============================================"
echo " SUCCESS!"
echo "============================================"
echo ""
echo "Sealed secret written to: $OUTPUT_FILE"
echo "Plain secret securely deleted."
echo ""
echo "Next steps:"
echo "  kubectl apply -f $OUTPUT_FILE"
echo ""
echo "To verify:"
echo "  kubectl get secret $SECRET_NAME -n $NAMESPACE -o jsonpath='{.data}' | jq 'keys'"
