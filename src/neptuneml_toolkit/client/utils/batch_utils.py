import boto3
from datetime import datetime
import time

from .cloudwatch_utils import print_logs

boto_session = boto3.DEFAULT_SESSION or boto3.Session()
region_name = boto_session.region_name

batch_log_group_name = '/aws/batch/job'

batch = boto3.client(
    service_name='batch',
    region_name='us-west-2',
    endpoint_url='https://batch.{region_name}.amazonaws.com'.format(region_name=region_name))

def wait_for_job(jobId):
    spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']
    spinner = 0
    running = False
    start_time = 0
    wait = True

    while wait:
        time.sleep(1)
        describeJobsResponse = batch.describe_jobs(jobs=[jobId])
        status = describeJobsResponse['jobs'][0]['status']
        if status == 'SUCCEEDED' or status == 'FAILED':
            print('%s' % ('=' * 80))
            print('Job [%s] %s' % (jobId, status))
            break
        elif status == 'RUNNING':
            log_stream_name = describeJobsResponse['jobs'][0]['container']['logStreamName']
            if not running and log_stream_name:
                running = True
                print('\rJob [%s] is RUNNING.' % (jobId), flush=True, end='')
                print('\nOutput [%s]:\n%s' % (log_stream_name, '=' * 80))
            if log_stream_name:
                start_time = print_logs(batch_log_group_name, log_stream_name, start_time) + 1
        else:
            print('\rJob [%s] is %-9s... %s' % (jobId, status, spin[spinner % len(spin)]), flush=True, end='')
            spinner += 1