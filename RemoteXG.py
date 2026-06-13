import base64
import json
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from mcp.server.fastmcp import FastMCP
import subprocess
import sys

# Initialize the FastMCP server
mcp = FastMCP("RemoteXG")

@mcp.tool()
def train_xgboost(
    X: list[list[float]], 
    y: list[float], 
    max_depth: int = 6, 
    learning_rate: float = 0.3, 
    n_estimators: int = 100, 
    objective: str = "reg:squarederror"
) -> str:
    """
    Trains an XGBoost model in-memory using provided datasets and hyperparameters.
    Returns metrics and a base64 serialized model artifact.
    """
    try:
        # 1. Map parameters directly from function arguments
        xgb_params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'objective': objective
        }

        # 2. Train/Test split in-memory (Avoiding DB completely)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 3. Instantiate and Train the model
        if "binary:" in xgb_params['objective']:
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metric_name = "accuracy"
            metric_value = float(accuracy_score(y_test, preds))
        else:
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metric_name = "rmse"
            metric_value = float(mean_squared_error(y_test, preds, squared=False))

        # 4. Serialize model directly to an ephemeral Base64 string
        serialized_model = base64.b64encode(pickle.dumps(model)).decode('utf-8')

        # 5. Build the response payload
        response_content = {
            "status": "success",
            "metrics": {
                "train_size": len(X_train),
                "test_size": len(X_test),
                metric_name: metric_value
            },
            "model_b64": serialized_model
        }

        return json.dumps(response_content)

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Training failed: {str(e)}"})

if __name__ == "__main__":
    # Allows Butterbase to run the server directly via standard CLI entrypoints
    mcp.run()
