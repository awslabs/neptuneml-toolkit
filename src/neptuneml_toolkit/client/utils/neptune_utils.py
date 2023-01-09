import boto3

try:
    boto_session = boto3.DEFAULT_SESSION or boto3.Session()
    region_name = boto_session.region_name
except:
    print("Could not get region from boto session. Use us-east-1 by default")
    region_name = 'us-east-1'

neptune_client = boto3.client(service_name='neptune', region_name=region_name)

def get_db_cluster_parameter_group(neptune_host):
    identifier = neptune_host.split(".")[0]
    describe_cluster_response = neptune_client.describe_db_clusters(DBClusterIdentifier=identifier)
    cluster_parameter_group = describe_cluster_response["DBClusters"][0]["DBClusterParameterGroup"]
    return cluster_parameter_group

def get_neptune_ml_iam_role_parameter_group(neptune_host):
    cluster_parameter_group_name = get_db_cluster_parameter_group(neptune_host)
    describe_parameter_group_response = neptune_client.describe_db_cluster_parameters(DBClusterParameterGroupName=cluster_parameter_group_name)
    if "Parameters" in describe_parameter_group_response:
        for parameter in describe_parameter_group_response["Parameters"]:
            if parameter["ParameterName"] == "neptune_ml_iam_role":
                if parameter["ParameterValue"]:
                    return parameter["ParameterValue"]
                else:
                    print("neptune_ml_iam_role Parameter not assigned")
        print("Could not find neptune_ml_iam_role parameter in cluster parameter group")
    raise