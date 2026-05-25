# ChatBI Agent - 绝配-港大AI赛 Question 1

A production-grade ChatBI agent that answers natural language questions about supply-chain order data by generating and executing pandas code using a local Ollama LLM.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place the two Excel data files in the `data/` folder:

```
chatbi_agent/
  data/
    测试客户下单量1月V2.xlsx
    测试客户下单2-3月V2.xlsx
```

### 3. Install Ollama & Pull Model

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from https://ollama.com/download

# Start Ollama server
ollama serve

# Pull the recommended model (in another terminal)
ollama pull qwen2.5-coder:7b
```

### 4. Run

```bash
# Single question
python main.py --question "1月20日当天有多少个配送订单被处理？"

# Interactive mode
python main.py --interactive

# Run QA tests
python main.py --test
```

## Architecture

```
Chinese Question
    |
    v
+----------------------------------+
| Schema + Few-Shot Prompt Builder |
+----------------------------------+
    |
    v
+------------------+     +------------------+
| Ollama Local LLM | --> | Pandas Code      |
| qwen2.5-coder:7b |     | (PAL pattern)    |
+------------------+     +------------------+
                              |
                              v
                       +------------------+
                       | Safe Sandbox     |
                       | Execution        |
                       +------------------+
                              |
                    +---------+---------+
                    |                   |
              Success             Error
                    |                   |
                    v                   v
            +------------+     +------------------+
            | Format NL  |     | Self-Correction  |
            | Answer     |     | (retry x2)       |
            +------------+     +------------------+
```

## File Structure

| File                    | Purpose                                      |
| ----------------------- | -------------------------------------------- |
| `config.py`           | Centralized configuration                    |
| `data_loader.py`      | Excel loading, column normalization, caching |
| `schema_manager.py`   | Schema extraction with Chinese descriptions  |
| `query_engine.py`     | Core PAL code generation engine              |
| `code_executor.py`    | Safe sandboxed execution with AST security   |
| `self_correction.py`  | Error classification and retry logic         |
| `answer_formatter.py` | Result to Chinese natural language           |
| `result_cache.py`     | Persistent query result caching              |
| `ollama_client.py`    | Ollama local LLM integration                 |
| `main.py`             | CLI entry point with all modes               |

## CLI Options

```
Mode Selection:
  --question, -q    Single question
  --batch, -b       Batch file (one question per line)
  --interactive, -i Interactive REPL
  --test, -t        Run built-in QA tests

Ollama Options:
  --ollama          Use Ollama local model (default)
  --no-ollama       Use cloud API instead
  --ollama-host     Ollama server URL
  --ollama-model    Model tag (auto-detect if omitted)
  --ollama-setup    Print setup guide

Cloud API Options:
  --api-key         OpenAI API key
  --base-url        API base URL
  --model           Model name

Other:
  --output, -o      Output file for results
  --no-cache        Disable caching
  --verbose, -v     Debug logging
  --profile         Show timing profile
```

## Competition Scoring Advantages

| Criteria       | Strategy                                           |
| -------------- | -------------------------------------------------- |
| Model Size     | 7B local (qwen2.5-coder:7b) = maximum bonus        |
| Query Speed    | Local inference, no network latency                |
| Accuracy       | PAL execution = exact numerical results            |
| Generalization | Schema-aware prompts handle any data-view question |

## Data Schema

### 订单表 (df_orders) - 30,815 rows

- 订单单号, 订单类型, 货主编码, 货主, 仓库, 收货门店, 省市区, 求和项:预计发货数量EA, 预计总箱数, 创建人, 创建时间

### 订单明细 (df_details) - 437,204 rows

- 订单单号, 商品编码, 商品名称, 温区, 预计发货数量, 单位, 预计发货数量EA, 单位.1

### 物流信息 (df_logistics) - 385,817 rows

- 订单号, 操作时间, 操作记录, 操作人

## License

Competition Submission

## Web GUI

Launch a beautiful chat interface in your browser:

```bash
python webapp.py
# Open http://localhost:5000
```

Features: chat-style UI, suggestion chips, code display, timing info
