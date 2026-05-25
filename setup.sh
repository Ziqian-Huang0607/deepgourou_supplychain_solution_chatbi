#!/bin/bash
# ChatBI Agent Setup Script
# =========================
# Run this script to quickly set up the environment for the 绝配-港大AI赛

set -e

echo "=========================================="
echo "ChatBI Agent - Setup"
echo "=========================================="

# 1. Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Python version: $PYTHON_VERSION"

# 2. Install Python dependencies
echo ""
echo "[1/4] Installing Python dependencies..."
pip install pandas numpy openpyxl openai

# 3. Create data directory
echo ""
echo "[2/4] Creating data directory..."
mkdir -p data
echo "Created: data/"
echo ""
echo "IMPORTANT: Please copy your Excel files into the data/ folder:"
echo "  cp /path/to/测试客户下单量1月V2.xlsx data/"
echo "  cp /path/to/测试客户下单2-3月V2.xlsx data/"

# 4. Check Ollama
echo ""
echo "[3/4] Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama found!"
    if ollama list &> /dev/null; then
        echo "Ollama is running."
        echo "Installed models:"
        ollama list
    else
        echo "Ollama is installed but not running."
        echo "Start it with: ollama serve"
    fi
    echo ""
    echo "Recommended: ollama pull qwen2.5-coder:7b"
else
    echo "Ollama not found. Install it:"
    echo "  macOS:  brew install ollama"
    echo "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or visit: https://ollama.com/download"
fi

# 5. Quick test
echo ""
echo "[4/4] Setup complete!"
echo ""
echo "=========================================="
echo "Quick Start Commands:"
echo "=========================================="
echo ""
echo "1. Place data files in data/ folder"
echo "2. Start Ollama:       ollama serve"
echo "3. Pull model:         ollama pull qwen2.5-coder:7b"
echo "4. Run single question:"
echo "   python main.py --question \"1月20日当天有多少个配送订单被处理？\""
echo ""
echo "5. Interactive mode:"
echo "   python main.py --interactive"
echo ""
echo "6. Run QA tests:"
echo "   python main.py --test"
echo ""
echo "7. Show Ollama setup guide:"
echo "   python main.py --ollama-setup"
echo ""
echo "=========================================="
