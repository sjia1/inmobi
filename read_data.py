from azure.storage.blob import BlobServiceClient
import numpy as np
import matplotlib.pyplot as plt 

# access data with shared access signature
account_name = 'inmobiai'
container = 'cmu'
account_url = "https://{}.blob.core.windows.net/".format(account_name)
blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)


# large number of files here
container_client = blob_service_client.get_container_client(container)
print("Listing blobs in container '{}' (size in bytes)".format(container))
blob_list = container_client.list_blobs()
for blob in blob_list:
    print("{:>10}  {}".format(blob.size, blob.name))
    
print("database accessed")


## Setting up Pandas
# read a single parquet file directly into a pandas dataframe using pyarrow
import pyarrow.parquet as pq
import io
import tempfile
import pandas as pd
#pd.set_option('max_columns', 200)
pd.set_option('max_columns', 500)
container_name = 'cmu'
print("pandas set")


## Interaction Dataset. Read just one parquet
parquet_file = 'glance/dataset-0/interaction-dataset/data/part-00000-0335229a-9d23-4ccc-8f79-8adb54fe4f73-c000.gz.parquet'
blob_client = container_client.get_blob_client(parquet_file)

# this will take some time...
with tempfile.TemporaryFile() as fp:
    blob_client.download_blob().download_to_stream(fp)
    action = pq.read_table(source=fp).to_pandas() # df
fp.close
print("interaction data loaded")


## User Metadata
tables=[]
blob_list = container_client.list_blobs()

for blob in blob_list:
    print(blob.name)
    if blob.name.startswith('glance/dataset-0/user-metadata/data/') and blob.name.endswith('.json'):
        blob_client = container_client.get_blob_client(blob.name)
        # this will take some time...
        # create temp files to download from Azure, can create permanent files too to avoid downloading again
        with tempfile.NamedTemporaryFile() as fp:
            blob_client.download_blob().download_to_stream(fp)
            tables.append(pd.read_json(fp.name, lines=True))
        fp.close
#         if len(tables) == 200: # only download the first 2 files to save memory and time; can be changed
#             break
user = pd.concat(tables)
del tables
print("user metadata loaded")


## Glance Metadata
json_file = 'glance/dataset-0/glance-metadata/data/part-00000-ecb4f2ad-5223-4913-8407-0606d75009b3-c000.json'
blob_client = container_client.get_blob_client(json_file)


# this will take some time...
with tempfile.NamedTemporaryFile() as fp:
    blob_client.download_blob().download_to_stream(fp)
    card = pd.read_json(fp.name, lines=True) # df
fp.close
print("card metadata loaded")


## Binge Session Data
tables=[]
blob_list = container_client.list_blobs()

for blob in blob_list:
    print(blob.name)
    if blob.name.startswith('glance/dataset-0/binge-sessions-impressions/data/') and blob.name.endswith('.json'):
        blob_client = container_client.get_blob_client(blob.name)
        # this will take some time...
        # create temp files to download from Azure, can create permanent files too to avoid downloading again
        with tempfile.NamedTemporaryFile() as fp:
            blob_client.download_blob().download_to_stream(fp)
            tables.append(pd.read_json(fp.name, lines=True))
        fp.close
        if len(tables) == 2: # only download the first 2 files to save memory and time; can be changed
            break
imp = pd.concat(tables)
del tables
print("imp metadata loaded")