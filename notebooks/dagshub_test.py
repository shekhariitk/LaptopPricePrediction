import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/shekhariitk/LaptopPricePrediction.mlflow")

dagshub.init(repo_owner='shekhariitk', repo_name='LaptopPricePrediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)