# Datapipeline
Here is what the updated pipeline would do.
```mermaid
graph LR
A[HuggingFace] -- 1. Download --> B{2. Preprocess on <br/> Virtual Machine}
B -- 3.Upload --> D[GCP Bucket]
```
## Step 1: Downloading
This step can be done using the `NWPPipeline.download` method. All we need to do is specify a path.
## Step 2: Preprocessing
This is where the bulk of work is done. In this step the data is preprocessed and wrangled. Here are most of the steps needed to transform the raw HuggingFace data into cleaned model ready data:
1.	Cropping region
2.	Cropping time
3.	Dropping unwanted features
4.	Interpolate?
5.	Other steps?
6.	Join PV
## Step 3: Upload
This step is easy, we already have scripts to upload a directory from local to a GCP bucket.
# Code Changes
- Rename `GCPPipeline` to `GCPPipelineUtils`. Treat it is as a general class for storing GCP pipeline related helper functions.
- Add functionality to interpolate and join PV data in `NWPPipeline`.
- Break down the `NWPPipeline.preprocess` method into several smaller composable methods that will come together to preprocess the data. An example:
	- `time_region_crop`
	- `drop_features`
	- `interpolate`
	- `join_with_pv`