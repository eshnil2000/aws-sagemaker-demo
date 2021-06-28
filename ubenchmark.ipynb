{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse\n",
    "import sklearn.datasets\n",
    "import pandas as pd\n",
    "\n",
    "outfile=\"data/data.csv\"\n",
    "X,y = sklearn.datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,)\n",
    "df = pd.DataFrame(X)\n",
    "df['Y'] = y\n",
    "df.to_csv(outfile, index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "train_instance_type has been renamed in sagemaker>=2.\n",
      "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.\n",
      "train_instance_type has been renamed in sagemaker>=2.\n",
      "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.\n"
     ]
    }
   ],
   "source": [
    "import argparse\n",
    "from sagemaker.sklearn import SKLearn\n",
    "import sagemaker\n",
    "from sagemaker import get_execution_role\n",
    "import boto3\n",
    "\n",
    "sagemaker_session = sagemaker.Session()\n",
    "\n",
    "role = get_execution_role()\n",
    "WORK_DIRECTORY = \"data\"\n",
    "# S3 prefix\n",
    "#s3://aws-sagemaker-demo\n",
    "bucket = sagemaker_session.default_bucket()\n",
    "\n",
    "prefix = \"aws-sagemaker-demo\"\n",
    "\n",
    "train_input = sagemaker_session.upload_data(\n",
    "    path=\"{}/{}\".format(WORK_DIRECTORY, \"data.csv\"),\n",
    "    bucket=bucket,\n",
    "    key_prefix=\"{}/{}\".format(prefix, \"data\"),\n",
    ")\n",
    "\n",
    "sklearn_estimator = SKLearn(\n",
    "        entry_point='src/train_and_deploy.py',\n",
    "        role=role,\n",
    "        train_instance_type='ml.m4.xlarge',\n",
    "        hyperparameters={\n",
    "            'sagemaker_submit_directory': f\"s3://{bucket}/{prefix}\",\n",
    "        },\n",
    "        framework_version='0.23-1',\n",
    "        metric_definitions=[\n",
    "            {'Name': 'train:score', 'Regex': 'train:score=(\\S+)'}],\n",
    "    )\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2021-06-18 03:14:30 Starting - Starting the training job...\n",
      "2021-06-18 03:14:56 Starting - Launching requested ML instancesProfilerReport-1623986070: InProgress\n",
      "......\n",
      "2021-06-18 03:15:56 Starting - Preparing the instances for training......\n",
      "2021-06-18 03:16:56 Downloading - Downloading input data...\n",
      "2021-06-18 03:17:16 Training - Downloading the training image...\n",
      "2021-06-18 03:17:59 Uploading - Uploading generated training model\n",
      "2021-06-18 03:17:59 Completed - Training job completed\n",
      "\u001b[34m2021-06-18 03:17:45,596 sagemaker-containers INFO     Imported framework sagemaker_sklearn_container.training\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:45,599 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:45,611 sagemaker_sklearn_container.training INFO     Invoking user training script.\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:45,991 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:46,217 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:46,232 sagemaker-training-toolkit INFO     No GPUs detected (normal if no gpus installed)\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:46,244 sagemaker-training-toolkit INFO     Invoking user script\n",
      "\u001b[0m\n",
      "\u001b[34mTraining Env:\n",
      "\u001b[0m\n",
      "\u001b[34m{\n",
      "    \"additional_framework_parameters\": {},\n",
      "    \"channel_input_dirs\": {\n",
      "        \"train\": \"/opt/ml/input/data/train\"\n",
      "    },\n",
      "    \"current_host\": \"algo-1\",\n",
      "    \"framework_module\": \"sagemaker_sklearn_container.training:main\",\n",
      "    \"hosts\": [\n",
      "        \"algo-1\"\n",
      "    ],\n",
      "    \"hyperparameters\": {},\n",
      "    \"input_config_dir\": \"/opt/ml/input/config\",\n",
      "    \"input_data_config\": {\n",
      "        \"train\": {\n",
      "            \"TrainingInputMode\": \"File\",\n",
      "            \"S3DistributionType\": \"FullyReplicated\",\n",
      "            \"RecordWrapperType\": \"None\"\n",
      "        }\n",
      "    },\n",
      "    \"input_dir\": \"/opt/ml/input\",\n",
      "    \"is_master\": true,\n",
      "    \"job_name\": \"sagemaker-scikit-learn-2021-06-18-03-14-30-470\",\n",
      "    \"log_level\": 20,\n",
      "    \"master_hostname\": \"algo-1\",\n",
      "    \"model_dir\": \"/opt/ml/model\",\n",
      "    \"module_dir\": \"s3://sagemaker-us-east-2-821921608777/sagemaker-scikit-learn-2021-06-18-03-14-30-470/source/sourcedir.tar.gz\",\n",
      "    \"module_name\": \"train_and_deploy\",\n",
      "    \"network_interface_name\": \"eth0\",\n",
      "    \"num_cpus\": 4,\n",
      "    \"num_gpus\": 0,\n",
      "    \"output_data_dir\": \"/opt/ml/output/data\",\n",
      "    \"output_dir\": \"/opt/ml/output\",\n",
      "    \"output_intermediate_dir\": \"/opt/ml/output/intermediate\",\n",
      "    \"resource_config\": {\n",
      "        \"current_host\": \"algo-1\",\n",
      "        \"hosts\": [\n",
      "            \"algo-1\"\n",
      "        ],\n",
      "        \"network_interface_name\": \"eth0\"\n",
      "    },\n",
      "    \"user_entry_point\": \"train_and_deploy.py\"\u001b[0m\n",
      "\u001b[34m}\n",
      "\u001b[0m\n",
      "\u001b[34mEnvironment variables:\n",
      "\u001b[0m\n",
      "\u001b[34mSM_HOSTS=[\"algo-1\"]\u001b[0m\n",
      "\u001b[34mSM_NETWORK_INTERFACE_NAME=eth0\u001b[0m\n",
      "\u001b[34mSM_HPS={}\u001b[0m\n",
      "\u001b[34mSM_USER_ENTRY_POINT=train_and_deploy.py\u001b[0m\n",
      "\u001b[34mSM_FRAMEWORK_PARAMS={}\u001b[0m\n",
      "\u001b[34mSM_RESOURCE_CONFIG={\"current_host\":\"algo-1\",\"hosts\":[\"algo-1\"],\"network_interface_name\":\"eth0\"}\u001b[0m\n",
      "\u001b[34mSM_INPUT_DATA_CONFIG={\"train\":{\"RecordWrapperType\":\"None\",\"S3DistributionType\":\"FullyReplicated\",\"TrainingInputMode\":\"File\"}}\u001b[0m\n",
      "\u001b[34mSM_OUTPUT_DATA_DIR=/opt/ml/output/data\u001b[0m\n",
      "\u001b[34mSM_CHANNELS=[\"train\"]\u001b[0m\n",
      "\u001b[34mSM_CURRENT_HOST=algo-1\u001b[0m\n",
      "\u001b[34mSM_MODULE_NAME=train_and_deploy\u001b[0m\n",
      "\u001b[34mSM_LOG_LEVEL=20\u001b[0m\n",
      "\u001b[34mSM_FRAMEWORK_MODULE=sagemaker_sklearn_container.training:main\u001b[0m\n",
      "\u001b[34mSM_INPUT_DIR=/opt/ml/input\u001b[0m\n",
      "\u001b[34mSM_INPUT_CONFIG_DIR=/opt/ml/input/config\u001b[0m\n",
      "\u001b[34mSM_OUTPUT_DIR=/opt/ml/output\u001b[0m\n",
      "\u001b[34mSM_NUM_CPUS=4\u001b[0m\n",
      "\u001b[34mSM_NUM_GPUS=0\u001b[0m\n",
      "\u001b[34mSM_MODEL_DIR=/opt/ml/model\u001b[0m\n",
      "\u001b[34mSM_MODULE_DIR=s3://sagemaker-us-east-2-821921608777/sagemaker-scikit-learn-2021-06-18-03-14-30-470/source/sourcedir.tar.gz\u001b[0m\n",
      "\u001b[34mSM_TRAINING_ENV={\"additional_framework_parameters\":{},\"channel_input_dirs\":{\"train\":\"/opt/ml/input/data/train\"},\"current_host\":\"algo-1\",\"framework_module\":\"sagemaker_sklearn_container.training:main\",\"hosts\":[\"algo-1\"],\"hyperparameters\":{},\"input_config_dir\":\"/opt/ml/input/config\",\"input_data_config\":{\"train\":{\"RecordWrapperType\":\"None\",\"S3DistributionType\":\"FullyReplicated\",\"TrainingInputMode\":\"File\"}},\"input_dir\":\"/opt/ml/input\",\"is_master\":true,\"job_name\":\"sagemaker-scikit-learn-2021-06-18-03-14-30-470\",\"log_level\":20,\"master_hostname\":\"algo-1\",\"model_dir\":\"/opt/ml/model\",\"module_dir\":\"s3://sagemaker-us-east-2-821921608777/sagemaker-scikit-learn-2021-06-18-03-14-30-470/source/sourcedir.tar.gz\",\"module_name\":\"train_and_deploy\",\"network_interface_name\":\"eth0\",\"num_cpus\":4,\"num_gpus\":0,\"output_data_dir\":\"/opt/ml/output/data\",\"output_dir\":\"/opt/ml/output\",\"output_intermediate_dir\":\"/opt/ml/output/intermediate\",\"resource_config\":{\"current_host\":\"algo-1\",\"hosts\":[\"algo-1\"],\"network_interface_name\":\"eth0\"},\"user_entry_point\":\"train_and_deploy.py\"}\u001b[0m\n",
      "\u001b[34mSM_USER_ARGS=[]\u001b[0m\n",
      "\u001b[34mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate\u001b[0m\n",
      "\u001b[34mSM_CHANNEL_TRAIN=/opt/ml/input/data/train\u001b[0m\n",
      "\u001b[34mPYTHONPATH=/opt/ml/code:/miniconda3/bin:/miniconda3/lib/python37.zip:/miniconda3/lib/python3.7:/miniconda3/lib/python3.7/lib-dynload:/miniconda3/lib/python3.7/site-packages\n",
      "\u001b[0m\n",
      "\u001b[34mInvoking script with the following command:\n",
      "\u001b[0m\n",
      "\u001b[34m/miniconda3/bin/python train_and_deploy.py\n",
      "\n",
      "\u001b[0m\n",
      "\u001b[34mtraining_files= []\u001b[0m\n",
      "\u001b[34mtrain:score=0.6949301962344134\u001b[0m\n",
      "\u001b[34m2021-06-18 03:17:48,161 sagemaker-containers INFO     Reporting training SUCCESS\u001b[0m\n",
      "Training seconds: 64\n",
      "Billable seconds: 64\n"
     ]
    }
   ],
   "source": [
    "# Run model training job\n",
    "sklearn_estimator.fit({\n",
    "        'train': f\"s3://{bucket}/{prefix}\"\n",
    "    })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "---------------!"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<sagemaker.sklearn.model.SKLearnPredictor at 0x7f0d330ad6d0>"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Deploy trained model to an endpoint\n",
    "endpoint_name='ubench-predict3'\n",
    "\n",
    "sklearn_estimator.deploy(\n",
    "        instance_type= 'ml.t2.medium',\n",
    "        initial_instance_count=1,\n",
    "        endpoint_name=endpoint_name,\n",
    "    )\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "response['Body']= b'[2.478031631449904, 2.478031631449904, 0.8173223265435456]'\n"
     ]
    }
   ],
   "source": [
    "client = boto3.client('sagemaker-runtime')\n",
    "response = client.invoke_endpoint(\n",
    "        EndpointName=endpoint_name,\n",
    "        Body=b\"[[1,2,3,4,5,6,7,8,9,1],[1,2,3,4,5,6,7,8,9,1],[100,1,1,1,1,1,1,1,1,1]]\",\n",
    "        ContentType='application/json',\n",
    "        Accept='application/json',\n",
    "    )\n",
    "print(\"response['Body']=\", response['Body'].read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 4 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "np.random.seed(19680801)\n",
    "data = np.random.randn(2, 100)\n",
    "\n",
    "fig, axs = plt.subplots(2, 2, figsize=(5, 5))\n",
    "axs[0, 0].hist(data[0])\n",
    "axs[1, 0].scatter(data[0], data[1])\n",
    "axs[0, 1].plot(data[0], data[1])\n",
    "axs[1, 1].hist2d(data[0], data[1])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "instance_type": "ml.t3.medium",
  "kernelspec": {
   "display_name": "Python 3 (SageMaker JumpStart Data Science 1.0)",
   "language": "python",
   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-east-2:033062077428:image/sagemaker-jumpstart-data-science-1.0"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
