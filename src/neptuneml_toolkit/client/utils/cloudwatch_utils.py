import boto3
from datetime import datetime

boto_session = boto3.DEFAULT_SESSION or boto3.Session()
region_name = boto_session.region_name
cloudwatch = boto3.client(
    service_name='logs',
    region_name=region_name,
    endpoint_url='https://logs.{region_name}.amazonaws.com'.format(region_name=region_name))

def get_log_stream_name(log_group_name, log_stream_prefix):
    kwargs = {'logGroupName': log_group_name,
              'logStreamNamePrefix': log_stream_prefix,
              'orderBy': 'LogStreamName',
              'descending': True}
    log_streams = cloudwatch.describe_log_streams(**kwargs)
    # assumes that only one log stream should be returned since neptune ml job names are unique and are not prefixes
    # of other job names
    if 'logStreams' in log_streams:
        if log_streams['logStreams']:
            return log_streams['logStreams'][0]['logStreamName']
    else:
        print('Log Stream not found for log group %s, and log stream prefix %s' % (log_group_name, log_stream_prefix))

    return None

def print_logs(log_group_name, log_stream_name, start_time):
    kwargs = {'logGroupName': log_group_name,
              'logStreamName': log_stream_name,
              'startTime': start_time,
              'startFromHead': True}

    lastTimestamp = start_time
    while True:
        logEvents = cloudwatch.get_log_events(**kwargs)
        nextToken = logEvents['nextForwardToken']
        events = logEvents['events']

        for event in events:
            lastTimestamp = event['timestamp']
            timestamp = datetime.utcfromtimestamp(lastTimestamp / 1000.0).isoformat()
            print('[%s] %s' % ((timestamp + ".000")[:23] + 'Z', event['message']))

        if nextToken and kwargs.get('nextToken') != nextToken:
            kwargs['nextToken'] = nextToken
        else:
            break
    return lastTimestamp