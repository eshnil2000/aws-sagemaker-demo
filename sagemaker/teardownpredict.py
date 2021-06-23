import boto3

deployment_name = 'my_deployment_name'
client = boto3.client('sagemaker')
response = client.describe_endpoint_config(EndpointConfigName=deployment_name)
model_name = response['ProductionVariants'][0]['ModelName']
client.delete_model(ModelName=model_name)    
client.delete_endpoint(EndpointName=deployment_name)
client.delete_endpoint_config(EndpointConfigName=deployment_name)
