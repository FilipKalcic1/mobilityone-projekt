"""
Test script to analyze the complete data flow from Swagger to extraction.
"""
import asyncio
import json
import httpx

async def analyze_swagger_flow():
    # Swagger URL
    base_url = 'https://m1aborapp.pokreni.dev'
    swagger_url = f'{base_url}/automation/swagger/v1.0.0/swagger.json'
    
    print(f'1. FETCHING SWAGGER: {swagger_url}')
    print('=' * 60)
    
    async with httpx.AsyncClient(verify=False, timeout=30) as client:
        response = await client.get(swagger_url)
        if response.status_code != 200:
            print(f'HTTP {response.status_code}')
            return
        
        spec = response.json()
        
        # Find MasterData endpoint
        paths = spec.get('paths', {})
        masterdata_path = paths.get('/MasterData', {})
        get_op = masterdata_path.get('get', {})
        
        print(f'\n2. /MasterData GET OPERATION')
        print('=' * 60)
        print(f'operationId: {get_op.get("operationId")}')
        print(f'tags: {get_op.get("tags")}')
        
        # Response schema
        responses = get_op.get('responses', {})
        success = responses.get('200', {})
        content = success.get('content', {})
        json_content = content.get('application/json', {}) or content.get('text/json', {})
        schema = json_content.get('schema', {})
        
        print(f'\n3. RESPONSE SCHEMA (raw)')
        print('=' * 60)
        print(json.dumps(schema, indent=2)[:500])
        
        # Resolve $ref
        def resolve_ref(ref_path, spec):
            parts = ref_path.replace('#/', '').split('/')
            resolved = spec
            for part in parts:
                resolved = resolved.get(part, {})
            return resolved
        
        if '$ref' in schema:
            ref_path = schema['$ref']
            print(f'\n4. RESOLVING $ref: {ref_path}')
            print('=' * 60)
            
            resolved = resolve_ref(ref_path, spec)
            props = resolved.get('properties', {})
            
            print(f'Schema type: {resolved.get("type")}')
            print(f'Number of properties: {len(props)}')
            print(f'\nALL PROPERTY NAMES FROM SWAGGER:')
            for i, name in enumerate(props.keys()):
                print(f'  {i+1}. {name}')
            
            # Check for important fields
            print(f'\n5. KEY FIELDS CHECK')
            print('=' * 60)
            important = ['LicencePlate', 'Plate', 'FullVehicleName', 'DisplayName', 
                        'Mileage', 'VIN', 'Driver', 'ProviderName', 'MonthlyAmount']
            for field in important:
                exists = field in props
                print(f'  {field}: {"✓ EXISTS" if exists else "✗ NOT IN SCHEMA"}')
        
        # Check if it's array type
        if schema.get('type') == 'array':
            items = schema.get('items', {})
            if '$ref' in items:
                resolved = resolve_ref(items['$ref'], spec)
                props = resolved.get('properties', {})
                print(f'\nArray items properties: {list(props.keys())[:10]}...')

if __name__ == '__main__':
    asyncio.run(analyze_swagger_flow())
