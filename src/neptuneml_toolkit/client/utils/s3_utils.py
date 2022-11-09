import json
import os
from sagemaker.s3 import S3Uploader, S3Downloader, s3_path_join

def upload(local_path, s3_location, kms_key=None):
    """
    Save a folder to an s3 location

    :param local_path: local path to upload
    :param s3_location: s3 location/folder to upload to
    :param kms_key: kms key to decrypt file
    :return:
    """
    return S3Uploader().upload(local_path, s3_location, kms_key=kms_key)

def download(s3_location, local_path, kms_key=None):
    """
    Download a folder from an s3 location

    :param s3_location: s3 location/folder to download
    :param local_path: local path to download to
    :param kms_key: kms key to decrypt file
    :return:
    """
    os.makedirs(local_directory, exist_ok=True)
    return S3Downloader().download(s3_location, local_directory, kms_key=kms_key)

def upload_config(config, s3_location, s3_file_name=None, kms_key=None):
    """
    Save a configuration document to an s3 location

    :param config: str/dict config json file path or configuration dict
    :param s3_location: s3 location/folder to upload config to
    :param s3_file_name: Name of the file in s3 folder. Required if config is a dict
    :param kms_key: kms key to decrypt file
    :return:
    """
    if type(config) == str:
        # assume it's a file path
        S3Uploader().upload(config, s3_location, kms_key=kms_key)
    else:
        assert type(config) == dict, "Configuration should be a local file or a python dictionary"
        assert s3_file_name is not None, "Configuration file name is required"
        config_str = json.dumps(config, indent=4)
        S3Uploader().upload_string_as_file_body(config_str, s3_path_join(s3_location, s3_file_name), kms_key=kms_key)


def get_config(s3_location, s3_file_name, kms_key=None, local_path=None):
    """
    Get a configuration document from s3

    :param s3_location: s3 location/folder to obtain config from
    :param s3_file_name: config file name at s3 location
    :param kms_key: kms key used to encrypt file
    :param local_path: local path to download configuration file to

    :return: configuration dict or None if local_path is passed
    """
    if local_path is not None:
        S3Downloader().download(s3_path_join(s3_location, s3_file_name), local_path, kms_key=kms_key)
    else:
        assert kms_key is None, "KMS encrypted files not supported for direct loading of configuration"
        json_str = S3Downloader().read_file(s3_path_join(s3_location, s3_file_name))
        return json.loads(json_str)