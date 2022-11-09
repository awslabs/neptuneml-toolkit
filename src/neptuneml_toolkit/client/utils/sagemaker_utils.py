import boto3
import time
import sys
import json
from .cloudwatch_utils import get_log_stream_name
import sagemaker

boto_session = boto3.DEFAULT_SESSION or boto3.Session()
sm_session = sagemaker.session.Session()
region_name = boto_session.region_name
sagemaker_client = boto3.client(
    service_name='sagemaker',
    region_name=region_name,
    endpoint_url='https://sagemaker.{region_name}.amazonaws.com'.format(region_name=region_name))

sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region_name)

log_group_name = "/aws/sagemaker/ProcessingJobs"
log_search_url = "https://{region_name}.console.aws.amazon.com/cloudwatch/home?region={region_name}#logStream:group={log_group_name};prefix={job_name}"

_STATUS_CODE_TABLE = {
    "COMPLETED": "Completed",
    "INPROGRESS": "InProgress",
    "FAILED": "Failed",
    "STOPPED": "Stopped",
    "STOPPING": "Stopping",
    "STARTING": "Starting",
    "CREATING": "Creating",
    "INSERVICE": "InService"
}

def wait_for_endpoint(endpoint_name):
    spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']
    spinner = 0
    start_time = 0
    wait = True

    while wait:
        time.sleep(1)
        describe_endpoint_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = describe_endpoint_response['EndpointStatus']
        if status == _STATUS_CODE_TABLE['INSERVICE']:
            print('\rEndpoint [%s] is %-9s... %s' % (endpoint_name, status, "!"),  flush=True, end='')
            break
        elif status == _STATUS_CODE_TABLE['FAILED']:
            print('%s' % ('=' * 80))
            print('Endpoint [%s] creation %s' % (endpoint_name, status))
            print('Failure Reason: %s' % (describe_endpoint_response['FailureReason']))
            break
        else:
            print('\rEndpoint [%s] is %-9s... %s' % (endpoint_name, status, spin[spinner % len(spin)]), flush=True, end='')
            spinner += 1

def wait_for_job(processing_job_name):
    spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']
    spinner = 0
    running = False
    start_time = 0
    wait = True
    log_stream_name = None

    while wait:
        time.sleep(1)
        describe_jobs_response = sagemaker_client.describe_processing_job(ProcessingJobName=processing_job_name)
        status = describe_jobs_response['ProcessingJobStatus']
        if status == _STATUS_CODE_TABLE['COMPLETED'] or status == _STATUS_CODE_TABLE['STOPPED']:
            print('%s' % ('=' * 80))
            print('Job [%s] %s' % (processing_job_name, status))
            break
        elif status == _STATUS_CODE_TABLE['FAILED']:
            print('%s' % ('=' * 80))
            print('Job [%s] %s' % (processing_job_name, status))
            print('Failure Reason: %s' % (describe_jobs_response['FailureReason']))
            print(log_search_url.format(region_name=region_name, log_group_name=log_group_name, job_name=processing_job_name))
            break
        elif status == _STATUS_CODE_TABLE['INPROGRESS']:
            if not log_stream_name:
                print('\rJob [%s] is %-9s... %s' % (processing_job_name, status, spin[spinner % len(spin)]), flush=True, end='')
                spinner += 1
                log_stream_name = get_log_stream_name(log_group_name, processing_job_name)
            if not running and log_stream_name:
                running = True
                print('\rJob [%s] is %-9s...' % (processing_job_name, status))
                print('Output [%s]:\n%s' % (log_stream_name, '=' * 80))
            if log_stream_name:
                sm_session.logs_for_processing_job(processing_job_name, wait=True)
        else:
            print('\rJob [%s] is %-9s... %s' % (processing_job_name, status, spin[spinner % len(spin)]), flush=True, end='')
            spinner += 1

def parse_arn_for_job_name(processing_job_arn):
    return processing_job_arn.split("/")[-1]

def get_processing_job_details(job_name):
    return sagemaker_client.describe_processing_job(ProcessingJobName=job_name)

def get_hyperparameter_tuning_job_details(job_name):
    return sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)

def invoke_endpoint(endpoint_name, data, content_type='application/json', response_format='application/json'):
    payload = json.dumps(data)
    response = sagemaker_runtime.invoke_endpoint(EndpointName=endpoint_name,
                                                 ContentType=content_type,
                                                 Accept=response_format,
                                                 Body=payload)
    return response['Body'].read().decode()