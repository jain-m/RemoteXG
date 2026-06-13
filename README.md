### RemoteXG 🚀

A stateless, production-ready Model Context Protocol (MCP) server that exposes distributed **XGBoost** training capabilities directly to LLMs, autonomous coding assistants, and local orchestration frameworks.

RemoteXG allows tools like Claude Code, Cursor, Windsurf, or custom agentic loops to instantly run optimized gradient-boosted tree architectures via standard protocols, completely decoupling heavy ML compute execution from context limits.

---

## Architecture Overview

```
 ┌────────────────────────────────────────────────────────┐
 │                      AI Client                         │
 │     (Claude Code, Cursor, Windsurf, MCP Inspector)     │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼  [ Standard I/O (stdio) Transport ]
 ┌────────────────────────────────────────────────────────┐
 │                    RemoteXG Server                     │
 │          (Python 3.11 + FastMCP Framework)             │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼  [ Parallel Machine Learning Compute ]
 ┌────────────────────────────────────────────────────────┐
 │                   XGBoost Core Engine                  │
 │          (libomp Multi-threaded Runtime)               │
 └────────────────────────────────────────────────────────┘

```

The system uses the **Model Context Protocol (MCP)** via standard input/output (`stdio`) streams. When an LLM client requests a training run, the server spins up a stateless local tracking context, trains an optimized native configuration on the input matrix data, and packages weights alongside key metrics back to the context window without saving heavy state artifacts to your filesystem.

---

## Prerequisite Configurations (macOS Installation)

XGBoost utilizes an internal parallel computing multi-threading architecture called **OpenMP** (`libomp`). Since Apple Silicon / macOS platforms do not ship with this compiler backend by default, you must configure it globally before running the tool:

1. **Install OpenMP Runtime via Homebrew:**
```bash
brew install libomp
```

2. **Install Required Python Dependencies:**
Install the core machine learning libraries along with the Model Context Protocol framework globally:

```bash
pip install xgboost scikit-learn mcp
```


3. **Verify Python Environment:**
Ensure you are using standard Homebrew or global Python 3.11 interpreters:
```bash
python3 --version
```



---

## Local Development & Debugging Workflow

To isolate code parsing errors without relying on production deployment resources, you can test the script interactively through the graphical web interface provided by the official **MCP Inspector**.

### Step 1: Initialize the MCP Inspector Interface

Run the network-isolated inspector loop in a clean terminal panel:

```bash
npx @modelcontextprotocol/inspector

```

This process will automatically build your testing dashboard and output the local web URL (typically `http://localhost:3000`).

### Step 2: Establish the Standard I/O Connection Bridge

Open the dashboard link in your browser. Inside the left-hand configuration sidebar, configure these exact transport instructions to safely spin up your local environment:

* **Transport Type:** `STDIO`
* **Command:** `env`
* **Arguments:** `python3 RemoteXG.py`

Click the dark blue **Connect** button. The interface will display a green **Connected** status marker and map out `RemoteXG` under the active server logs block.

### Step 3: Interactive Verification Test Case

Navigate to the **Tools** tab at the top of the interface. Locate the `train_xgboost` tool schema, input these dummy arrays into the fields, and verify execution:

* **`X` (JSON Matrix):** `[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]`
* **`y` (JSON Vector):** `[1.5, 2.5, 3.5, 4.5, 5.5]`
* **`max_depth`:** `3`
* **`learning_rate`:** `0.3`
* **`n_estimators`:** `100`
* **`objective`:** `reg:squarederror`

Click **Run Tool**. A successful execution returns a raw structured text response highlighting computed loss values and a compiled Base64 binary serialization format of your newly generated booster architecture.

---

## Transitioning to Live Production (Butterbase Integration)

Once local execution parameters have been stabilized inside the browser interface, your stateless structure is ready to be transitioned out into high-performance cloud hosting via **Butterbase**.

### Step 1: Acquire your Production Credentials

Log into your project console at [butterbase.ai/dashboard](https://www.google.com/search?q=https://butterbase.ai/dashboard) and navigate to the **API Keys** section. Capture your unique master authorization token (`bb_sk_...`). You will need this key if you are configuring editor-level extensions like Cursor, Claude Code, or Windsurf to authenticate global background processes against your account.

### Step 2: Log Into Your Butterbase Workspace via CLI

For deployment and direct invocation testing, authenticate your terminal session natively to bypass end-user JWT requirements:

```bash
npx @butterbase/cli login

```

Follow the terminal prompts to paste your API token or complete the terminal handshake.

### Step 3: Provision Your Serverless Infrastructure

Deploy the project bundle directly to your host workspace. By targeting your edge-optimized JavaScript handler, the system processes your files and builds the serverless function mapping:

```bash
npx @butterbase/cli functions deploy index.js

```

Upon a successful upload build string, the CLI will output your live invocation path:

```
✔ Function deployed successfully!
  Invoke URL: /v1/app_chc5aqphyxmx/fn/index

```

### Step 4: Register RemoteXG to your Global `mcp.json` Config

To hook the newly containerized cloud endpoint into tools like **Claude Desktop**, append the server configuration parameters directly into your local configuration file (typically mapped at `~/Library/Application Support/Claude/mcp.json`):

```json
{
  "mcpServers": {
    "remotexg-prod": {
      "command": "npx",
      "args": [
        "@butterbase/cli",
        "functions",
        "invoke",
        "index"
      ]
    }
  }
}
```

### Step 5: Perform a Live End-to-End Test

Verify that your active workspace session seamlessly signs your payloads by executing the live production endpoint directly using the session-aware CLI utility:

```bash
npx @butterbase/cli functions invoke index --data '{
  "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
  "y": [1.5, 2.5, 3.5, 4.5, 5.5],
  "max_depth": 3,
  "learning_rate": 0.3,
  "n_estimators": 10
}' | jq -r '.model_b64' | base64 -d > models/model.json
```

A valid execution handshake will return your structured decision tree matrices and final training loss metrics directly in your console! 🚀