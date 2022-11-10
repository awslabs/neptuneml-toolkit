import boto3
import os
from copy import copy
from botocore.session import get_session
from graph_notebook.neptune.client import Client, ClientBuilder, NEPTUNE_CONFIG_HOST_IDENTIFIERS, \
    is_allowed_neptune_host
from graph_notebook.configuration.generate_config import (generate_default_config, DEFAULT_CONFIG_LOCATION,
                                                          AuthModeEnum, Configuration)
from graph_notebook.configuration.get_config import get_config
from urllib.parse import urlparse

from .utils import s3_utils, sagemaker_utils, batch_utils, neptune_utils

import numpy as np
import pickle
import time

DEFAULT_REGION = boto3.Session().region_name
DEFAULT_PROCESSING_CONFIG_FILE_NAME = 'training-data-configuration.json'
DEFAULT_TRAINING_CONFIG_FILE_NAME = 'model-hpo-configuration.json'
DEFAULT_EMBEDDING_FILE_NAME = 'entity.npy'
DEFAULT_PREDICTION_FILE_NAME = 'result.npz'
DEFAULT_TRAINING_MAPPING_INFO = 'mapping.info'
DEFAULT_PROCESSING_MAPPING_INFO = 'info.pkl'
DEFAULT_TRAINING_MAPPING_KEY = 'node2id'
DEFAULT_PROCESSING_MAPPING_KEY = 'node_id_map'
DEFAULT_ENTITY_MAPPING_KEY = 'node2gid'
DEFAULT_MAX_HPO_NUM = 2
DEFAULT_MAX_PARALLEL_HPO = 2
DEFAULT_EVAL_METRIC_FILE = "eval_metrics_info.json"
DEFAULT_TEST_METRIC_FILE = "test_metrics_info.json"
DEFAULT_TRAIN_INSTANCE_REC = "train_instance_recommendation.json"
DEFAULT_INFER_INSTANCE_REC = "infer_instance_recommendation.json"


class NeptuneMLClient():

    def __init__(self, host=None, port=8182, auth_mode='DEFAULT', use_ssl=False, region=DEFAULT_REGION,
                 load_from_s3_arn='', proxy_host='', proxy_port=8182, neptune_hosts=NEPTUNE_CONFIG_HOST_IDENTIFIERS,
                 config_location=DEFAULT_CONFIG_LOCATION, neptune_iam_role_arn='', sagemaker_iam_role_arn='',
                 subnets=None, security_group_ids=None, volume_encryption_kms_key=None,
                 s3_output_encryption_kms_key=None,
                 export_service_uri=None):
        """
        Creates a NeptuneMLClient for interacting NeptuneML model management APIs

        :param host:str the host url to form a connection with.
        :param port:int the port to use when creating a connection. Default 8182
        :param auth_mode:str type of authentication the cluster being connected to is using. Can be DEFAULT or IAM
        :param use_ssl:bool whether to make connections to the created endpoint with ssl or not. Default True
        :param region:str aws region your Neptune cluster is in. Default
        :param load_from_s3_arn:str arn of role to use for bulk loader.
        :param proxy_host:str the proxy host url to route a connection through.
        :param proxy_port:int the proxy port to use when creating proxy connection. Default 8182
        :param neptune_hosts:list of host snippets to use for identifying neptune endpoints. Default ["amazonaws.com"]
        :param config_location:str location of existing Neptune db client config. Default '~/graph_notebook_config.json'
        :param neptune_iam_role_arn:str arn of role used by neptune to for neptune ml commands
        :param sagemaker_iam_role_arn:str arn of role to use within sagemaker jobs
        :param subnets:
        :param security_group_ids:
        :param volume_encryption_kms_key:
        :param s3_output_encryption_kms_key:
        :param export_service_uri: url of Neptune export service
        """
        self.client = None
        self.neptune_cfg_allowlist = copy(neptune_hosts)

        if host is not None:
            auth_mode = auth_mode if auth_mode in [auth.value for auth in AuthModeEnum] else AuthModeEnum.DEFAULT.value
            config = generate_config(host, port, AuthModeEnum(auth_mode), use_ssl, load_from_s3_arn, region,
                                     proxy_host, proxy_port, neptune_hosts=neptune_hosts)
        else:
            self.config_location = config_location or os.getenv('GRAPH_NOTEBOOK_CONFIG', DEFAULT_CONFIG_LOCATION)
            config = get_config(self.config_location, neptune_hosts=self.neptune_cfg_allowlist)

        self._config = config
        self._generate_client_from_config(config)

        if not neptune_iam_role_arn:
            neptune_iam_role_arn = neptune_utils.get_neptune_ml_iam_role_parameter_group(self.get_host())

        self._register_security_params(sagemaker_iam_role_arn, neptune_iam_role_arn, subnets, security_group_ids,
                                       volume_encryption_kms_key, s3_output_encryption_kms_key)

        self._export_service_host = None
        if export_service_uri is not None:
            path = urlparse(export_service_uri.rstrip())
            self._export_service_host = path.hostname + "/v1"
        else:
            self._export_service_host = self.get_export_service_host()
            assert self._export_service_host is not None, "Export service is required for Neptune ML"

    def _generate_client_from_config(self, config):
        if self.client:
            self.client.close()

        is_neptune_host = is_allowed_neptune_host(hostname=config.host, host_allowlist=self.neptune_cfg_allowlist)
        assert is_neptune_host, "Neptune ML only works with Amazon NeptuneDB"

        self.builder = ClientBuilder() \
            .with_host(config.host) \
            .with_port(config.port) \
            .with_region(config.aws_region) \
            .with_tls(config.ssl) \
            .with_proxy_host(config.proxy_host) \
            .with_proxy_port(config.proxy_port) \
            .with_sparql_path(config.sparql.path)
        if config.auth_mode == AuthModeEnum.IAM:
            self.builder = self.builder.with_iam(get_session())
        if self.neptune_cfg_allowlist != NEPTUNE_CONFIG_HOST_IDENTIFIERS:
            self.builder = self.builder.with_custom_neptune_hosts(self.neptune_cfg_allowlist)

        self.client = self.builder.build()

    def _register_security_params(self, sagemaker_iam_role_arn, neptune_iam_role_arn, subnets, security_group_ids,
                                  volume_encryption_kms_key, s3_output_encryption_kms_key):
        self.sagemaker_iam_role_arn = sagemaker_iam_role_arn
        self.neptune_iam_role_arn = neptune_iam_role_arn
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.volume_encryption_kms_key = volume_encryption_kms_key
        self.s3_output_encryption_kms_key = s3_output_encryption_kms_key

    def check_enabled(self):
        try:
            assert self.get_host() is not None
            data_processing_job = self.list_data_processing_jobs()
            assert self.get_export_service_host() is not None
            neptune_utils.get_neptune_ml_iam_role_parameter_group(self.get_host())
            print("This Neptune cluster is configured to use Neptune ML")
        except Exception as ex:
            print(ex)
            print('''This Neptune cluster \033[1mis not\033[0m configured to use Neptune ML.
                Please configure the cluster according to the Amazon Neptune ML documentation before proceeding.''')

    def get_export_service_host(self):
        if self._export_service_host is not None:
            return self._export_service_host
        try:
            with open('/home/ec2-user/.bashrc') as f:
                data = f.readlines()
            for d in data:
                if str.startswith(d, 'export NEPTUNE_EXPORT_API_URI'):
                    parts = d.split('=')
                    if len(parts) == 2:
                        path = urlparse(parts[1].rstrip())
                        return path.hostname + "/v1"
        except:
            print("Unable to determine the Neptune Export Service Endpoint. You will need to this assign this manually")
        return None

    def get_host(self):
        if self._config is not None:
            return self._config.host
        else:
            return None

    def get_port(self):
        if self._config is not None:
            return self._config.port
        else:
            return None

    def get_iam(self):
        if self._config is not None:
            return self._config.auth_mode == AuthModeEnum.IAM
        else:
            return None

    def _add_security_params(self, params, sagemaker_iam_role_arn, neptune_iam_role_arn, subnets, security_group_ids,
                             volume_encryption_kms_key, s3_output_encryption_kms_key):

        sagemaker_iam_role_arn = sagemaker_iam_role_arn or self.sagemaker_iam_role_arn
        if sagemaker_iam_role_arn:
            params['sagemakerIamRoleArn'] = sagemaker_iam_role_arn
        neptune_iam_role_arn = neptune_iam_role_arn or self.neptune_iam_role_arn
        if neptune_iam_role_arn:
            params['neptuneIamRoleArn'] = neptune_iam_role_arn
        subnets = subnets or self.subnets
        if subnets:
            if all([isinstance(subnet, str) for subnet in subnets]):
                params['subnets'] = subnets
            else:
                print('Ignoring subnets, list does not contain all strings.')
        security_group_ids = security_group_ids or self.security_group_ids
        if security_group_ids:
            if all([isinstance(security_group_id, str) for security_group_id in security_group_ids]):
                params['securityGroupIds'] = security_group_ids
            else:
                print('Ignoring security group IDs, list does not contain all strings.')
        volume_encryption_kms_key = volume_encryption_kms_key or self.volume_encryption_kms_key
        if volume_encryption_kms_key:
            params['volumeEncryptionKMSKey'] = volume_encryption_kms_key
        s3_output_encryption_kms_key = s3_output_encryption_kms_key or self.s3_output_encryption_kms_key
        if s3_output_encryption_kms_key:
            params['s3OutputEncryptionKMSKey'] = s3_output_encryption_kms_key

        return params

    def get_timestamp_job_name(self, prefix):
        return f'{prefix}-{int(time.time())}'

    def create_data_export_job(self,
                               command=None,
                               outputS3Path=None,
                               jobSize=None,
                               exportProcessParams=None,
                               additionalParams=None,
                               export_url=None,
                               export_iam=True,
                               ssl=True,
                               wait=False,
                               params={}):
        if params:
            assert "command" in params, "'command' must be in params and be one of export_pg or export_rdf"
            assert "outputS3Path" in params, "'outputS3Path' must be in params"
        else:
            assert outputS3Path is not None, "outputS3path is required"
            params = {}
            params['outputS3Path'] = outputS3Path
            if command is not None:
                params['command'] = command
            if jobSize is not None:
                params['jobSize'] = jobSize
            if exportProcessParams is not None:
                params['params'] = exportProcessParams
            if additionalParams is not None:
                params['additionalParams'] = additionalParams

        if export_iam:
            self.builder = self.builder.with_iam(get_session())
        export_client = self.builder.build()

        export_url = export_url or self.get_export_service_host()
        export_job_response = export_client.export(export_url, params, ssl)
        export_job_response.raise_for_status()
        export_job = export_job_response.json()
        if wait == True:
            batch_utils.wait_for_job(export_job["jobId"])
            return self.describe_data_export_job(export_job["jobId"])
        return export_job

    def describe_data_export_job(self, id, export_url=None, export_iam=True, ssl=True):
        if export_iam:
            self.builder = self.builder.with_iam(get_session())
        export_client = self.builder.build()

        export_url = export_url or self.get_export_service_host()
        export_status = export_client.export_status(export_url, id, ssl)
        export_status.raise_for_status()
        export_job = export_status.json()
        return export_job

    def get_training_data_configuration(self, export_job_id):
        export_job = self.describe_data_export_job(export_job_id)
        return s3_utils.get_config(export_job['outputS3Uri'], DEFAULT_PROCESSING_CONFIG_FILE_NAME)

    def create_data_processing_job(self,
                                   id=None,
                                   previousDataProcessingJobId=None,
                                   inputDataS3Location=None,
                                   processedDataS3Location=None,
                                   sagemakerIamRoleArn=None,
                                   neptuneIamRoleArn=None,
                                   processingInstanceType=None,
                                   processingInstanceVolumeSizeInGB=None,
                                   processingTimeOutInSeconds=None,
                                   modelType=None,
                                   configFileName=None,
                                   subnets=None,
                                   securityGroupIds=None,
                                   volumeEncryptionKMSKey=None,
                                   s3OutputEncryptionKMSKey=None,
                                   trainingDataConfiguration=None,
                                   params={},
                                   wait=False):

        if trainingDataConfiguration is not None:
            if type(trainingDataConfiguration) == str:
                assert os.path.exists(
                    trainingDataConfiguration), "trainingDataConfiguration file not found at path {}".format(
                    trainingDataConfiguration)
                s3_utils.upload_config(trainingDataConfiguration, inputDataS3Location)
            else:
                assert type(trainingDataConfiguration) == dict
                configFileName = configFileName or params.get('configFileName', DEFAULT_PROCESSING_CONFIG_FILE_NAME)
                s3_utils.upload_config(trainingDataConfiguration, inputDataS3Location, s3_file_name=configFileName)

        if params:
            assert 'inputDataS3Location' in params or inputDataS3Location, "inputDataS3Location must be provided"
            assert 'processedDatas3Location' in params or processedDataS3Location, "processedDataS3Location must be provided"
            inputDataS3Location = inputDataS3Location or params['inputDataS3Location']
            processedDataS3Location = processedDataS3Location or params['processedDataS3Location']
        else:
            assert inputDataS3Location is not None, "inputDataS3Location must be provided"
            assert processedDataS3Location is not None, "processedDataS3Location must be provided"
            params = {}
            if id:
                params['id'] = id
            if previousDataProcessingJobId:
                params['previousDataProcessingJobId'] = previousDataProcessingJobId
            if modelType:
                params['modelType'] = modelType
            if configFileName:
                params['configFileName'] = configFileName
            self._add_security_params(params, sagemakerIamRoleArn, neptuneIamRoleArn, subnets, securityGroupIds,
                                      volumeEncryptionKMSKey, s3OutputEncryptionKMSKey)

        data_processing_job_response = self.client.dataprocessing_start(inputDataS3Location, processedDataS3Location,
                                                                        **params)
        data_processing_job_response.raise_for_status()
        data_processing_job = data_processing_job_response.json()
        if wait:
            neptuneIamRoleArn = params.get("neptuneIamRoleArn", neptuneIamRoleArn)
            data_processing_status = self.describe_data_processing_job(data_processing_job["id"],
                                                                       neptuneIamRoleArn=neptuneIamRoleArn)
            sagemaker_utils.wait_for_job(data_processing_status["processingJob"]["name"])

        return data_processing_job

    def describe_data_processing_job(self, id, neptuneIamRoleArn=None, verbose=False):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            data_processing_status = self.client.dataprocessing_job_status(id, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            data_processing_status = self.client.dataprocessing_job_status(id)

        data_processing_status.raise_for_status()
        data_processing_job = data_processing_status.json()

        if verbose:
            data_processing_job["processingJobDetails"] = sagemaker_utils.get_processing_job_details(
                data_processing_job["processingJob"]["name"])

        return data_processing_job

    def list_data_processing_jobs(self, maxItems=10, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            list_result = self.client.dataprocessing_list(max_items=maxItems, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            list_result = self.client.dataprocessing_list(max_items=maxItems)
        list_result.raise_for_status()
        return list_result.json()

    def stop_data_processing_job(self, id, clean=False, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            stop_result = self.client.modeltransform_stop(id, clean=clean, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            stop_result = self.client.modeltransform_stop(id, clean=clean)
        stop_result.raise_for_status()

    def get_model_hpo_configuration(self, data_processing_job_id):
        data_processing_job = self.describe_data_processing_job(data_processing_job_id)
        return s3_utils.get_config(data_processing_job["processingJob"]["outputLocation"],
                                   DEFAULT_TRAINING_CONFIG_FILE_NAME)

    def get_model_training_instance_recommendation(self, data_processing_job_id):
        data_processing_job = self.describe_data_processing_job(data_processing_job_id)
        return s3_utils.get_config(data_processing_job["processingJob"]["outputLocation"], DEFAULT_TRAIN_INSTANCE_REC)

    def create_model_training_job(self,
                                  id=None,
                                  dataProcessingJobId=None,
                                  trainModelS3Location=None,
                                  previousModelTrainingJobId=None,
                                  sagemakerIamRoleArn=None,
                                  neptuneIamRoleArn=None,
                                  modelName=None,
                                  baseProcessingInstanceType=None,
                                  trainingInstanceType=None,
                                  trainingInstanceVolumeSizeInGB=None,
                                  trainingTimeOutInSeconds=None,
                                  maxHPONumberOfTrainingJobs=None,
                                  maxHPOParallelTrainingJobs=None,
                                  subnets=None,
                                  securityGroupIds=None,
                                  volumeEncryptionKMSKey=None,
                                  s3OutputEncryptionKMSKey=None,
                                  enableManagedSpotTraining=None,
                                  customModelTrainingParameters=None,
                                  modelHPOConfiguration=None,
                                  params={},
                                  wait=False):
        assert dataProcessingJobId is not None or "dataProcessingJobId" in params, "dataProcessingId is required"
        assert trainModelS3Location is not None or "trainModelS3ocation" in params, "trainModelS3Location is required"
        if modelHPOConfiguration is not None:
            s3_train_input_uri = self.describe_data_processing_job(data_processing_job["id"],
                                                                   neptuneIamRoleArn=neptuneIamRoleArn)[
                "processingJob"]["outputLocation"]
            if type(modelHPOConfiguration) == str:
                assert os.path.exists(
                    modelHPOConfiguration), "trainingDataConfiguration file not found at path {}".format(
                    trainingDataConfiguration)
                s3_utils.upload_config(modelHPOConfiguration, s3_train_input_uri)
            else:
                assert type(trainingDataConfiguration) == dict
                s3_utils.upload_config(modelHPOConfiguration, s3_train_input_uri,
                                       s3_file_name=DEFAULT_TRAINING_CONFIG_FILE_NAME)

        if customModelTrainingParameters is not None:
            customModelTrainingParameters = dict(customModelTrainingParameters)
            assert 'sourceS3DirectoryPath' in customModelTrainingParameters, "sourceS3DirectoryPath must be provided in customModelTrainingParameters"
            if 'sourceDirectory' in customModelTrainingParameters:
                source_directory = customModelTrainingParameters.pop('sourceDirectory')
                s3_utils.upload(source_directory, customModelTrainingParameters['sourceS3DirectoryPath'])

        if params:
            dataProcessingJobId = dataProcessingJobId or params['dataProcessingJobId']
            trainModelS3Location = trainModelS3Location or params['trainModelS3Location']
            maxHPONumberOfTrainingJobs = maxHPONumberOfTrainingJobs or params['maxHPONumberOfTrainingJobs']
            maxHPOParallelTrainingJobs = maxHPOParallelTrainingJobs or params['maxHPOParallelTrainingJobs']
        else:
            maxHPONumberOfTrainingJobs = maxHPONumberOfTrainingJobs or DEFAULT_MAX_HPO_NUM
            maxHPOParallelTrainingJobs = maxHPOParallelTrainingJobs or DEFAULT_MAX_PARALLEL_HPO
            params = {}
            if id:
                params['id'] = id
            if previousModelTrainingJobId:
                params['previousModelTrainingJobId'] = previousModelTrainingJobId
            if modelName:
                params['modelName'] = modelName
            if baseProcessingInstanceType:
                params['baseProcessingInstanceType'] = baseProcessingInstanceType
            if trainingInstanceType:
                params['trainingInstanceType'] = trainingInstanceType
            if trainingInstanceVolumeSizeInGB:
                params['trainingInstanceVolumeSizeInGB'] = trainingInstanceVolumeSizeInGB
            if trainingTimeOutInSeconds:
                params['trainingTimeOutInSeconds'] = trainingTimeOutInSeconds
            if enableManagedSpotTraining:
                params['enableManagedSpotTraining'] = enableManagedSpotTraining
            if customModelTrainingParameters:
                params['customModelTrainingParameters'] = customModelTrainingParameters

            self._add_security_params(params, sagemakerIamRoleArn, neptuneIamRoleArn, subnets, securityGroupIds,
                                      volumeEncryptionKMSKey, s3OutputEncryptionKMSKey)

        model_training_job_response = self.client.modeltraining_start(dataProcessingJobId, trainModelS3Location,
                                                                      maxHPONumberOfTrainingJobs,
                                                                      maxHPOParallelTrainingJobs, **params)
        model_training_job_response.raise_for_status()
        model_training_job = model_training_job_response.json()
        if wait:
            neptuneIamRoleArn = params.get("neptuneIamRoleArn", neptuneIamRoleArn)
            model_training_status = self.describe_model_training_job(model_training_job["id"],
                                                                     neptuneIamRoleArn=neptuneIamRoleArn)
            sagemaker_utils.wait_for_job(model_training_status["processingJob"]["name"])

        return model_training_job

    def describe_model_training_job(self, id, neptuneIamRoleArn=None, verbose=False):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            model_training_status = self.client.modeltraining_job_status(id, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            model_training_status = self.client.modeltraining_job_status(id)

        model_training_status.raise_for_status()
        model_training_job = model_training_status.json()

        if verbose:
            model_training_job["processingJobDetails"] = sagemaker_utils.get_processing_job_details(
                model_training_job["processingJob"]["name"])
            model_training_job[
                "hyperparameterTuningJobDetails"] = sagemaker_utils.get_hyperparameter_tuning_job_details(
                model_training_job["hpoJob"]["name"])

        return model_training_job

    def list_model_training_jobs(self, maxItems=10, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            list_result = self.client.modeltraining_list(max_items=maxItems, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            list_result = self.client.modeltraining_list(max_items=maxItems)
        list_result.raise_for_status()
        return list_result.json()

    def stop_model_training_job(self, id, clean=False, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            stop_result = self.client.modeltraining_stop(id, clean=clean, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            stop_result = self.client.modeltraining_stop(id, clean=clean)
        stop_result.raise_for_status()

    def create_model_transform_job(self,
                                   id=None,
                                   dataProcessingJobId=None,
                                   mlModelTrainingJobId=None,
                                   trainingJobName=None,
                                   modelTransformOutputS3Location=None,
                                   sagemakerIamRoleArn=None,
                                   neptuneIamRoleArn=None,
                                   modelName=None,
                                   baseProcessingInstanceType=None,
                                   baseProcessingInstanceVolumeSizeInGB=None,
                                   subnets=None,
                                   securityGroupIds=None,
                                   volumeEncryptionKMSKey=None,
                                   s3OutputEncryptionKMSKey=None,
                                   customModelTransformParameters=None,
                                   params={},
                                   wait=False):
        dataProcessingJobId = dataProcessingJobId or params.get('dataProcessingJobId', None)
        mlModelTrainingJobId = mlModelTrainingJobId or params.get('mlModelTrainingJobId', None)
        trainingJobName = trainingJobName or params.get('trainingJobName', None)

        assert (dataProcessingJobId is not None and mlModelTrainingJobId is not None) or trainingJobName is not None, \
            "Both dataProcessingJobId and mlModelTrainingJobId are required or trainingJobName is required"

        modelTransformOutputS3Location = modelTransformOutputS3Location or params.get('modelTransformOutputS3Location',
                                                                                      None)
        assert modelTransformOutputS3Location is not None, "modelTransformOutputS3Location is required"

        if customModelTransformParameters is not None:
            customModelTransformParameters = dict(customModelTransformParameters)
            assert 'sourceS3DirectoryPath' in customModelTrainingParameters, "sourceS3DirectoryPath must be provided in customModelTrainingParameters"
            if 'sourceDirectory' in customModelTrainingParameters:
                source_directory = customModelTrainingParameters.pop('sourceDirectory')
                s3_utils.upload(source_directory, customModelTrainingParameters['sourceS3DirectoryPath'])

        if not params:
            params = {}
            if id:
                params['id'] = id
            if baseProcessingInstanceType:
                params['baseProcessingInstanceType'] = baseProcessingInstanceType
            if baseProcessingInstanceVolumeSizeInGB:
                params['trainingInstanceVolumeSizeInGB'] = trainingInstanceVolumeSizeInGB
            if customModelTransformParameters:
                params['customModelTransformParameters'] = customModelTransformParameters

            self._add_security_params(params, sagemakerIamRoleArn, neptuneIamRoleArn, subnets, securityGroupIds,
                                      volumeEncryptionKMSKey, s3OutputEncryptionKMSKey)

        model_transform_job_response = self.client.modeltransform_create(modelTransformOutputS3Location,
                                                                         dataProcessingJobId,
                                                                         mlModelTrainingJobId,
                                                                         trainingJobName, **params)
        model_transform_job_response.raise_for_status()
        model_transform_job = model_transform_job_response.json()
        if wait:
            neptuneIamRoleArn = params.get("neptuneIamRoleArn", neptuneIamRoleArn)
            model_transform_status = self.describe_model_transform_job(model_transform_job["id"],
                                                                       neptuneIamRoleArn=neptuneIamRoleArn)
            sagmaker_utils.wait_for_job(model_transform_status["processingJob"]["name"])

        return model_transform_job

    def describe_model_transform_job(self, id, neptuneIamRoleArn=None, verbose=False):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            model_transform_status = self.client.modeltransform_job_status(id, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            model_transform_status = self.client.modeltransform_job_status(id)

        model_transform_status.raise_for_status()
        model_transform_job = model_transform_status.json()

        if verbose:
            model_transform_job["processingJobDetails"] = sagemaker_utils.get_processing_job_details(
                model_transform_job["processingJob"]["name"])

        return model_transform_job

    def list_model_transform_jobs(self, maxItems=10, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            list_result = self.client.modeltransform_list(max_items=maxItems, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            list_result = self.client.modeltransform_list(max_items=maxItems)
        list_result.raise_for_status()
        return list_result.json()

    def stop_model_transform_job(self, id, clean=False, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            stop_result = self.client.modeltransform_stop(id, clean=clean, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            stop_result = self.client.modeltransform_stop(id, clean=clean)
        stop_result.raise_for_status()

    def create_endpoint(self,
                        id=None,
                        mlModelTrainingJobId=None,
                        mlModelTransformJobId=None,
                        update=False,
                        modelName=None,
                        neptuneIamRoleArn=None,
                        instanceType=None,
                        instanceCount=None,
                        volumeEncryptionKMSKey=None,
                        params={},
                        wait=False):
        mlModelTrainingJobId = mlModelTrainingJobId or params.get('mlModelTrainingJobId', None)
        mlModelTransformJobId = mlModelTransformJobId or params.get('mlModelTransformJobId', None)

        assert (mlModelTrainingJobId is not None or mlModelTransformJobId is not None), \
            "Either mlModelTrainingJobId or mlModelTransformJobId is required"

        if not params:
            params = {}
            if id:
                params['id'] = id
            if update:
                params['update'] = update
            if modelName:
                params['modelName'] = modelName
            if instanceType:
                params['instanceType'] = instanceType
            if instanceCount:
                params['instanceCount'] = instanceCount
            if neptuneIamRoleArn:
                params['neptuneIamRoleArn'] = neptuneIamRoleArn
            if volumeEncryptionKMSKey:
                params['volumeEncryptionKMSKey'] = volumeEncryptionKMSKey

        endpoint_response = self.client.endpoints_create(mlModelTrainingJobId, mlModelTransformJobId, **params)
        endpoint_response.raise_for_status()
        endpoint = endpoint_response.json()
        if wait:
            neptuneIamRoleArn = params.get("neptuneIamRoleArn", neptuneIamRoleArn)
            endpoint_status = self.describe_endpoint(endpoint["id"], neptuneIamRoleArn=neptuneIamRoleArn)
            sagemaker_utils.wait_for_endpoint(endpoint_status["endpoint"]["name"])

        return endpoint

    def update_endpoint(self,
                        id=None,
                        mlModelTrainingJobId=None,
                        mlModelTransformJobId=None,
                        modelName=None,
                        instanceType=None,
                        instanceCount=None,
                        volumeEncryptionKMSKey=None,
                        params={},
                        wait=False):
        return self.create_endpoint(id=id,
                                    mlModelTrainingJobId=mlModelTrainingJobId,
                                    mlModelTransformJobId=mlModelTransformJobId,
                                    modelName=modelName,
                                    instanceType=instanceType,
                                    instanceCount=instanceCount,
                                    volumeEncryptionKMSKey=volumeEncryptionKMSKey,
                                    params=params,
                                    wait=wait,
                                    update=True)

    def describe_endpoint(self, id, neptuneIamRoleArn=None, verbose=False):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            endpoint_status = self.client.endpoints_status(id, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            endpoint_status = self.client.endpoints_status(id)

        endpoint_status.raise_for_status()
        endpoint = endpoint_status.json()

        return endpoint

    def list_endpoints(self, maxItems=10, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            list_result = self.client.endpoints(max_items=maxItems, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            list_result = self.client.endpoints(max_items=maxItems)
        list_result.raise_for_status()
        return list_result.json()

    def delete_endpoint(self, id, neptuneIamRoleArn=None):
        neptuneIamRoleArn = neptuneIamRoleArn or self.neptune_iam_role_arn
        if neptuneIamRoleArn is not None:
            delete_result = self.client.endpoints_delete(id, neptune_iam_role_arn=neptuneIamRoleArn)
        else:
            delete_result = self.client.endpoints_delete(id)
        delete_result.raise_for_status()

    def get_node_index_mapping(self, data_processing_job_id=None, vertex_label=None, download_location='./model-artifacts'):
        assert data_processing_job_id is not None, \
            "You must provide either a data processing job id to obtain node to index mappings"

        job_details = self.describe_data_processing_job(data_processing_job_id)
        job_s3_output = job_details["processingJob"]["outputLocation"]

        # get mappings
        download_location = os.path.join(download_location, data_processing_job_id)
        s3_utils.download(os.path.join(job_s3_output, DEFAULT_PROCESSING_MAPPING_INFO), download_location)

        with open(os.path.join(download_location, DEFAULT_PROCESSING_MAPPING_INFO), "rb") as f:
            info = pickle.load(f)
            mapping = info[DEFAULT_PROCESSING_MAPPING_KEY]
            embedding_index_mapping = None
            if vertex_label is not None:
                if vertex_label in mapping:
                    mapping = mapping[vertex_label]
                else:
                    print("Mapping for vertex label: {} not found.".format(vertex_label))
                    print("valid vertex labels which have vertices mapped to embeddings: {} ".format(
                        list(mapping.keys())))
                    print("Returning mapping for all valid vertex labels")
        return mapping, embedding_index_mapping

    def get_embeddings(self, model_training_job_id, download_location='./model-artifacts', kms_key=None):
        assert model_training_job_id is not None, "model_training_job_id is required"
        training_job_s3_output = self.describe_model_training_job(model_training_job_id)["processingJob"][
            "outputLocation"]
        download_directory = os.path.join(download_location, model_training_job_id, "embeddings")

        s3_utils.download(os.path.join(training_job_s3_output, "embeddings"), download_directory, kms_key=kms_key)
        entity_emb = np.load(os.path.join(download_directory, DEFAULT_EMBEDDING_FILE_NAME))

        return entity_emb

    def get_predictions(self, model_training_job_id, download_location='./model-artifacts', class_preds=False,
                        kms_key=None):
        assert model_training_job_id is not None, "model_training_job_id is required"
        training_job_s3_output = self.describe_model_training_job(model_training_job_id)["processingJob"][
            "outputLocation"]
        download_directory = os.path.join(download_location, model_training_job_id, "predictions")

        s3_utils.download(os.path.join(training_job_s3_output, "predictions"), download_directory, kms_key)
        preds = np.load(os.path.join(download_directory, DEFAULT_PREDICTION_FILE_NAME))['infer_scores']

        if class_preds:
            return preds.argmax(axis=1)

        return preds

    def get_model_performance_metrics(self, model_training_job_id):
        assert model_training_job_id is not None, "model_training_job_id is required"
        training_job_s3_output = self.describe_model_training_job(model_training_job_id)["processingJob"][
            "outputLocation"]
        metrics = {}
        metrics['validation'] = s3_utils.get_config(training_job_s3_output, DEFAULT_EVAL_METRIC_FILE)
        try:
            metrics['test'] = s3_utils.get_config(training_job_s3_output, DEFAULT_TEST_METRIC_FILE)
        except:
            print("Test metrics not present, Returning only validation metrics")
        return metrics

    def get_endpoint_instance_recommendation(self, model_training_job_id=None, model_transform_job_id=None):
        assert model_training_job_id is not None or model_transform_job_id is not None, \
            "You must provide either a model training job id or a model transform job id to get endpoint instance recommendation"

        job_id = model_training_job_id if model_training_job_id is not None else model_transform_job_id
        job_details = self.describe_model_training_job(
            job_id) if job_id == model_training_job_id else self.model_transform_job_id(job_id)
        job_s3_output = job_details["processingJob"]["outputLocation"]
        return s3_utils.get_config(job_s3_output, DEFAULT_INFER_INSTANCE_REC)

    def invoke_endpoint(self, task="link_predict", endpoint_id=None, endpoint_name=None, headNodeId=None,
                        tailNodeId=None,
                        headNodeType=None, tailNodeType=None, edgeType=None, property=None, topk=1):
        if endpoint_name is None:
            assert endpoint_id is not None, "endpoint id is a required argument if endpoint name is missing"
            endpoint_name = self.describe_endpoint(endpoint_id)["endpoint"]["name"]
        if task in ["link_predict", "link_prediction"]:
            task = "link_predict"
            assert edgeType is not None, "edgeType is required for link prediction"
            if headNodeType is not None and tailNodeId is not None and edgeType is not None:
                assert headNodeId is None, "headNodeId should not be be passed for link prediction when tailNodeId is passed"
            elif tailNodeType is not None and headNodeId is not None and edgeType is not None:
                assert tailNodeId is None, "tailNodeId should not be be passed for link prediction when headNodeId is passed"
            else:
                print(
                    "For link prediction you must pass the pair (headNodeType, tailNodeId) or (tailNodeType , headNodeid)")
                raise RuntimeError
        elif task in ["edge_predict", "edge_prediction", "edge_property_prediction", "edge_classification",
                      "edge_regression"]:
            task = "edge_predict"
            assert headNodeId is not None and tailNodeId is not None, "headNodeId and tailNodeId is required for edge prediction"
            assert edgeType is not None, "edgeType should be passed for edge prediction"
            assert property is not None, "property should be passed in for edge prediction"

        elif task in ["node_predict", "node_prediction", "node_property_prediction", "node_classification",
                      "node_regression"]:
            task = "edge_predict"
            assert tailNodeId is None, "tailNodeId should not be be passed for node prediction."
            assert headNodeId is not None, "headNodeId should be passed in for node prediction"
            assert property is not None, "property should be passed in for node prediction"
        elif task in ["knn_predict", "knn_prediction"]:
            task = "knn_predict"
            assert tailNodeId is None, "tailNodeId should not be be passed for knn prediction."
            assert headNodeId is not None, "headNodeId should be passed in for knn prediction"
            assert tailNodeType is not None, "tailNodeType is required for knn prediction"
        else:
            print("Unsupported task type : {}".format(task))
            raise RuntimeError

        input_data = {}
        input_data["version"] = "v1"
        if task == "link_predict":
            if headNodeId is not None:
                input_data["mode"] = "predict_tail"
                input_data["data"] = {
                    "globalParameters": {
                        "topk": topk,
                        "edgeType": edgeType,
                        "tailNodeType": tailNodeType,
                        "exclude_flag": 'mask'
                    },
                    "edges": [
                        {
                            "headNodeId": headNodeId
                        }
                    ]
                }
            else:
                input_data["mode"] = "predict_head"
                input_data["data"] = {
                    "globalParameters": {
                        "topk": topk,
                        "edgeType": edgeType,
                        "headNodeType": headNodeType,
                        "exclude_flag": 'mask'
                    },
                    "edges": [
                        {
                            "tailNodeId": tailNodeId
                        }
                    ]
                }
        elif task == "edge_predict":
            input_data["data"] = {
                "globalParameters": {
                    "topk": topk,
                    "edgeType": edgeType,
                    "property": property
                },
                "edges": [
                    {
                        "headNodeId": headNodeId,
                        "tailNodeId": tailNodeId
                    }
                ]
            }
        elif task == "node_predict":
            input_data["data"] = {
                "globalParameters": {
                    "topk": topk,
                    "property": property
                },
                "nodes": [
                    {
                        "nodeId": headNodeId
                    }
                ]
            }
        elif task == "knn_predict":
            input_data["mode"] = "embed_knn"
            input_data["data"] = {
                "globalParameters": {
                    "topk": topk,
                    "tailNodeType": tailNodeType,
                    "exclude_flag": 'mask'
                },
                "nodes": [
                    {
                        "nodeId": headNodeId
                    }
                ]
            }
        else:
            print("Unsupported task type : {}".format(task))
            raise RuntimeError

        return sagemaker_utils.invoke_endpoint(endpoint_name, input_data)