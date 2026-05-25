#!/usr/bin/env python3
"""
ChatBI Agent - ChatGPT-style Web GUI
A clean, minimal chat interface for natural language to pandas code generation.

Usage:
    export OLLAMA_MODEL=qwen2.5-coder:7b
    python webapp.py
    # Opens http://localhost:5000
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template_string, request, jsonify

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbi-web")

# ---------------------------------------------------------------------------
# Ensure local modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from main import ChatBIAgent
except ImportError as e:
    logger.warning("Could not import ChatBIAgent from main.py: %s", e)
    ChatBIAgent = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PORT = int(os.environ.get("CHATBI_PORT", "5000"))
HOST = os.environ.get("CHATBI_HOST", "0.0.0.0")

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "chatbi-dev-secret-key")

# ---------------------------------------------------------------------------
# Agent singleton
# ---------------------------------------------------------------------------
_agent = None


def get_agent():
    """Lazy-initialize the ChatBIAgent."""
    global _agent
    if _agent is None and ChatBIAgent is not None:
        logger.info("Initializing ChatBIAgent with model=%s", OLLAMA_MODEL)
        try:
            _agent = ChatBIAgent(ollama_model=OLLAMA_MODEL, ollama_host=OLLAMA_URL)
        except Exception as e:
            logger.error("Failed to initialize agent: %s", e)
    return _agent


# ---------------------------------------------------------------------------
# HTML Template (complete, self-contained)
# ---------------------------------------------------------------------------
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatBI Agent</title>
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🤖%3C/text%3E%3C/svg%3E">
    <style>
        /* ── CSS Reset & Base ── */
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f9f9f9;
            --bg-sidebar: #f9f9f9;
            --bg-chat: #ffffff;
            --bg-user-bubble: #f7f7f8;
            --bg-bot-bubble: #ffffff;
            --bg-code: #1e1e1e;
            --bg-code-header: #2d2d2d;
            --bg-input: #ffffff;
            --bg-hover: #ececf1;
            --bg-selected: #e3e3e3;
            --text-primary: #343541;
            --text-secondary: #6e6e80;
            --text-muted: #8e8ea0;
            --border-color: #e5e5e5;
            --accent: #10a37f;
            --accent-hover: #0d8c6d;
            --accent-light: #e6f5f1;
            --error: #ef4444;
            --error-bg: #fef2f2;
            --shadow: 0 1px 3px rgba(0,0,0,0.04);
            --shadow-card: 0 2px 12px rgba(0,0,0,0.06);
            --font-main: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: "SF Mono", "Fira Code", "Cascadia Code", "Source Code Pro", Menlo, Monaco, Consolas, monospace;
            --radius: 8px;
            --radius-lg: 12px;
            --sidebar-width: 260px;
            --header-height: 44px;
            --transition: all 0.2s ease;
        }

        [data-theme="dark"] {
            --bg-primary: #343541;
            --bg-secondary: #444654;
            --bg-sidebar: #202123;
            --bg-chat: #343541;
            --bg-user-bubble: #444654;
            --bg-bot-bubble: #343541;
            --bg-code: #1e1e1e;
            --bg-code-header: #2d2d2d;
            --bg-input: #40414f;
            --bg-hover: #3e3f4b;
            --bg-selected: #353740;
            --text-primary: #ececf1;
            --text-secondary: #c5c5d2;
            --text-muted: #8e8ea0;
            --border-color: #4d4d4f;
            --accent: #19c59f;
            --accent-hover: #14a882;
            --accent-light: rgba(25,197,159,0.15);
            --error: #f87171;
            --error-bg: rgba(239,68,68,0.1);
            --shadow: 0 1px 3px rgba(0,0,0,0.2);
            --shadow-card: 0 2px 12px rgba(0,0,0,0.3);
        }

        html, body {
            height: 100%;
            font-family: var(--font-main);
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--bg-primary);
            overflow: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* ── Layout ── */
        .app {
            display: flex;
            height: 100vh;
            width: 100vw;
        }

        /* ── Sidebar ── */
        .sidebar {
            width: var(--sidebar-width);
            min-width: var(--sidebar-width);
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            transition: var(--transition);
        }

        .sidebar-header {
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .new-chat-btn {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            background: transparent;
            color: var(--text-primary);
            font-size: 14px;
            font-family: var(--font-main);
            cursor: pointer;
            transition: var(--transition);
        }

        .new-chat-btn:hover {
            background: var(--bg-hover);
        }

        .new-chat-btn svg {
            width: 16px;
            height: 16px;
            flex-shrink: 0;
        }

        .theme-toggle {
            width: 36px;
            height: 36px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            flex-shrink: 0;
        }

        .theme-toggle:hover {
            color: var(--text-primary);
            background: var(--bg-hover);
        }

        .theme-toggle svg {
            width: 16px;
            height: 16px;
        }

        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }

        .sidebar-content::-webkit-scrollbar { width: 4px; }
        .sidebar-content::-webkit-scrollbar-track { background: transparent; }
        .sidebar-content::-webkit-scrollbar-thumb { background: transparent; border-radius: 2px; }
        .sidebar-content:hover::-webkit-scrollbar-thumb { background: var(--border-color); }

        .conversation-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: var(--radius);
            cursor: pointer;
            font-size: 14px;
            color: var(--text-secondary);
            transition: var(--transition);
            margin-bottom: 2px;
            position: relative;
        }

        .conversation-item:hover {
            background: var(--bg-hover);
        }

        .conversation-item.active {
            background: var(--bg-selected);
            color: var(--text-primary);
        }

        .conversation-item svg {
            width: 16px;
            height: 16px;
            flex-shrink: 0;
        }

        .conversation-title {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .sidebar-footer {
            padding: 12px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
        }

        /* ── Main Content ── */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-chat);
            overflow: hidden;
            position: relative;
        }

        /* ── Header ── */
        .header {
            height: var(--header-height);
            min-height: var(--header-height);
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid var(--border-color);
            position: relative;
            z-index: 10;
        }

        .model-info {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--accent);
            display: inline-block;
        }

        .status-dot.offline {
            background: var(--error);
        }

        .menu-toggle {
            display: none;
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            width: 32px;
            height: 32px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            align-items: center;
            justify-content: center;
        }

        /* ── Chat Area ── */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            scroll-behavior: smooth;
        }

        .chat-area::-webkit-scrollbar { width: 6px; }
        .chat-area::-webkit-scrollbar-track { background: transparent; }
        .chat-area::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }

        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
            text-align: center;
        }

        .welcome-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }

        .welcome-title {
            font-size: 28px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .welcome-subtitle {
            font-size: 15px;
            color: var(--text-secondary);
            margin-bottom: 32px;
            max-width: 420px;
        }

        .suggestion-chips {
            display: flex;
            flex-direction: column;
            gap: 8px;
            width: 100%;
            max-width: 480px;
        }

        .suggestion-chip {
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            background: var(--bg-primary);
            color: var(--text-primary);
            font-size: 14px;
            font-family: var(--font-main);
            text-align: left;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: var(--shadow);
        }

        .suggestion-chip:hover {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }

        .suggestion-chip-title {
            font-weight: 500;
            margin-bottom: 2px;
        }

        .suggestion-chip-desc {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* ── Messages ── */
        .messages {
            display: flex;
            flex-direction: column;
        }

        .message-row {
            display: flex;
            padding: 20px;
            border-bottom: 1px solid transparent;
        }

        .message-row.user {
            background: var(--bg-user-bubble);
        }

        .message-row.bot {
            background: var(--bg-bot-bubble);
        }

        .message-row.bot + .message-row.bot,
        .message-row.user + .message-row.user {
            border-top: none;
        }

        .message-container {
            display: flex;
            gap: 16px;
            max-width: 800px;
            width: 100%;
            margin: 0 auto;
            padding: 0 8px;
        }

        .avatar {
            width: 30px;
            height: 30px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .avatar.user {
            background: #5436da;
            color: white;
            order: 2;
        }

        .avatar.bot {
            background: var(--accent);
            color: white;
        }

        .message-content {
            flex: 1;
            min-width: 0;
            color: var(--text-primary);
            font-size: 15px;
            line-height: 1.7;
            word-wrap: break-word;
        }

        .message-content p { margin-bottom: 0.7em; }
        .message-content p:last-child { margin-bottom: 0; }
        .message-content pre { margin: 0; }
        .message-content ul, .message-content ol {
            margin: 0.5em 0 0.5em 1.5em;
        }
        .message-content li { margin-bottom: 0.3em; }
        .message-content strong { font-weight: 600; }
        .message-content code {
            font-family: var(--font-mono);
            font-size: 0.88em;
            padding: 2px 5px;
            border-radius: 4px;
            background: var(--bg-hover);
        }
        .message-content a { color: var(--accent); text-decoration: none; }
        .message-content a:hover { text-decoration: underline; }
        .message-content table {
            border-collapse: collapse;
            margin: 0.8em 0;
            font-size: 13px;
        }
        .message-content th, .message-content td {
            border: 1px solid var(--border-color);
            padding: 6px 10px;
        }
        .message-content th {
            background: var(--bg-hover);
            font-weight: 600;
        }

        /* ── Code Blocks ── */
        .code-block-wrapper {
            margin: 12px 0;
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .code-block-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 14px;
            background: var(--bg-code-header);
            color: #d4d4d4;
            font-size: 12px;
            font-family: var(--font-mono);
        }

        .code-lang-tag {
            text-transform: lowercase;
            color: #858585;
        }

        .code-actions {
            display: flex;
            gap: 6px;
        }

        .code-action-btn {
            background: transparent;
            border: none;
            color: #858585;
            cursor: pointer;
            font-size: 12px;
            font-family: var(--font-main);
            padding: 3px 8px;
            border-radius: 4px;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .code-action-btn:hover {
            color: #d4d4d4;
            background: rgba(255,255,255,0.08);
        }

        .code-action-btn svg {
            width: 12px;
            height: 12px;
        }

        .code-block-wrapper pre {
            background: var(--bg-code) !important;
            padding: 14px 16px;
            overflow-x: auto;
            font-family: var(--font-mono);
            font-size: 13px;
            line-height: 1.6;
            color: #d4d4d4;
        }

        .code-block-wrapper pre code {
            background: none !important;
            padding: 0 !important;
            font-family: var(--font-mono);
        }

        /* Syntax Highlighting Colors */
        .code-block-wrapper .kw { color: #569cd6; }   /* keyword */
        .code-block-wrapper .str { color: #ce9178; }   /* string */
        .code-block-wrapper .num { color: #b5cea8; }   /* number */
        .code-block-wrapper .cmt { color: #6a9955; }   /* comment */
        .code-block-wrapper .fn { color: #dcdcaa; }    /* function */
        .code-block-wrapper .cls { color: #4ec9b0; }   /* class */

        /* ── Timing Info ── */
        .timing-info {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .message-row:hover .timing-info {
            opacity: 1;
        }

        /* ── Typing Indicator ── */
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 8px 0;
        }

        .typing-indicator span {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--text-muted);
            animation: typingBounce 1.4s infinite ease-in-out;
        }

        .typing-indicator span:nth-child(1) { animation-delay: 0s; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typingBounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-6px); }
        }

        /* ── Input Area ── */
        .input-area {
            border-top: 1px solid var(--border-color);
            background: var(--bg-chat);
            padding: 16px 16px 20px;
        }

        .input-container {
            max-width: 800px;
            margin: 0 auto;
            position: relative;
        }

        .input-box-wrapper {
            display: flex;
            align-items: flex-end;
            gap: 8px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            background: var(--bg-input);
            padding: 10px 14px;
            box-shadow: var(--shadow-card);
            transition: var(--transition);
        }

        .input-box-wrapper:focus-within {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent), var(--shadow-card);
        }

        .input-box {
            flex: 1;
            border: none;
            outline: none;
            background: transparent;
            font-family: var(--font-main);
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-primary);
            resize: none;
            max-height: 200px;
            min-height: 24px;
            padding: 2px 0;
        }

        .input-box::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            width: 32px;
            height: 32px;
            border: none;
            border-radius: 8px;
            background: var(--accent);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: var(--transition);
        }

        .send-btn:hover {
            background: var(--accent-hover);
        }

        .send-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .send-btn svg {
            width: 16px;
            height: 16px;
        }

        .input-hint {
            text-align: center;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 6px;
        }

        /* ── Error Banner ── */
        .error-banner {
            background: var(--error-bg);
            border: 1px solid var(--error);
            border-radius: var(--radius);
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 13px;
            color: var(--error);
        }

        /* ── Loading Overlay ── */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .spinner {
            width: 36px;
            height: 36px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Toast Notification ── */
        .toast {
            position: fixed;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%) translateY(20px);
            background: var(--text-primary);
            color: var(--bg-primary);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 13px;
            opacity: 0;
            transition: all 0.3s ease;
            z-index: 200;
            pointer-events: none;
        }

        .toast.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }

        /* ── Scroll to bottom button ── */
        .scroll-to-bottom {
            position: absolute;
            bottom: 80px;
            left: 50%;
            transform: translateX(-50%);
            width: 36px;
            height: 36px;
            border: 1px solid var(--border-color);
            border-radius: 50%;
            background: var(--bg-primary);
            color: var(--text-muted);
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-card);
            z-index: 5;
            transition: var(--transition);
        }

        .scroll-to-bottom.visible {
            display: flex;
        }

        .scroll-to-bottom:hover {
            color: var(--text-primary);
            background: var(--bg-hover);
        }

        .scroll-to-bottom svg {
            width: 16px;
            height: 16px;
        }

        /* ── Responsive ── */
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                left: -100%;
                top: 0;
                bottom: 0;
                z-index: 50;
                box-shadow: 2px 0 8px rgba(0,0,0,0.15);
            }
            .sidebar.open { left: 0; }
            .menu-toggle { display: flex; }
            .message-container { padding: 0; }
            .message-row { padding: 14px 12px; }
            .welcome-title { font-size: 22px; }
            .input-area { padding: 10px 10px 14px; }
        }

        /* ── Fade-in animation ── */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.3s ease forwards;
        }
    </style>
</head>
<body>
    <div class="app">
        <!-- Sidebar -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <button class="new-chat-btn" onclick="startNewChat()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19"></line>
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                    </svg>
                    New chat
                </button>
                <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark mode">
                    <svg id="theme-icon-light" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                    </svg>
                    <svg id="theme-icon-dark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;">
                        <circle cx="12" cy="12" r="5"></circle>
                        <line x1="12" y1="1" x2="12" y2="3"></line>
                        <line x1="12" y1="21" x2="12" y2="23"></line>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                        <line x1="1" y1="12" x2="3" y2="12"></line>
                        <line x1="21" y1="12" x2="23" y2="12"></line>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                    </svg>
                </button>
            </div>
            <div class="sidebar-content" id="conversation-list">
                <!-- Populated by JS -->
            </div>
            <div class="sidebar-footer">
                ChatBI Agent v1.0 &middot; Local LLM
            </div>
        </aside>

        <!-- Main -->
        <div class="main">
            <!-- Header -->
            <header class="header">
                <button class="menu-toggle" id="menu-toggle" onclick="toggleSidebar()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="3" y1="12" x2="21" y2="12"></line>
                        <line x1="3" y1="6" x2="21" y2="6"></line>
                        <line x1="3" y1="18" x2="21" y2="18"></line>
                    </svg>
                </button>
                <div class="model-info">
                    <span class="status-dot" id="status-dot"></span>
                    <span id="model-name">{{ model }}</span>
                </div>
            </header>

            <!-- Chat Area -->
            <div class="chat-area" id="chat-area">
                <div class="welcome-screen" id="welcome-screen">
                    <div class="welcome-icon">🤖</div>
                    <div class="welcome-title">ChatBI Agent</div>
                    <div class="welcome-subtitle">
                        Ask me anything about your data in natural language. I'll generate pandas code and give you the answer.
                    </div>
                    <div class="suggestion-chips" id="suggestion-chips">
                        <button class="suggestion-chip" onclick="sendSuggestion(this)">
                            <div class="suggestion-chip-title">📊 Show me a summary</div>
                            <div class="suggestion-chip-desc">"Give me a summary of the dataset"</div>
                        </button>
                        <button class="suggestion-chip" onclick="sendSuggestion(this)">
                            <div class="suggestion-chip-title">🔍 Filter and analyze</div>
                            <div class="suggestion-chip-desc">"Show rows where sales > 1000, sorted by date"</div>
                        </button>
                        <button class="suggestion-chip" onclick="sendSuggestion(this)">
                            <div class="suggestion-chip-title">📈 Statistics</div>
                            <div class="suggestion-chip-desc">"What is the average revenue by category?"</div>
                        </button>
                        <button class="suggestion-chip" onclick="sendSuggestion(this)">
                            <div class="suggestion-chip-title">🔗 Group and aggregate</div>
                            <div class="suggestion-chip-desc">"Group by region and sum the profit"</div>
                        </button>
                    </div>
                </div>
                <div class="messages" id="messages" style="display:none;"></div>
            </div>

            <!-- Scroll to bottom -->
            <button class="scroll-to-bottom" id="scroll-to-bottom" onclick="scrollToBottom()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </button>

            <!-- Input Area -->
            <div class="input-area">
                <div class="input-container">
                    <div class="input-box-wrapper">
                        <textarea
                            class="input-box"
                            id="user-input"
                            placeholder="Ask about your data..."
                            rows="1"
                            oninput="autoResize(this)"
                            onkeydown="handleKeyDown(event)"
                        ></textarea>
                        <button class="send-btn" id="send-btn" onclick="sendMessage()" title="Send message">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        </button>
                    </div>
                    <div class="input-hint">Shift + Enter for new line &middot; Enter to send</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast -->
    <div class="toast" id="toast"></div>

    <script>
        // ── State ──
        let conversations = [];
        let currentConversationId = null;
        let isWaiting = false;

        const MODEL = "{{ model }}";
        const el = id => document.getElementById(id);

        // ── Theme ──
        function initTheme() {
            const saved = localStorage.getItem('chatbi-theme');
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const theme = saved || (prefersDark ? 'dark' : 'light');
            document.documentElement.setAttribute('data-theme', theme);
            updateThemeIcon(theme);
        }

        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('chatbi-theme', next);
            updateThemeIcon(next);
        }

        function updateThemeIcon(theme) {
            el('theme-icon-light').style.display = theme === 'dark' ? 'none' : 'block';
            el('theme-icon-dark').style.display = theme === 'dark' ? 'block' : 'none';
        }

        // ── Conversations ──
        function loadConversations() {
            const saved = localStorage.getItem('chatbi-conversations');
            if (saved) {
                conversations = JSON.parse(saved);
                renderConversationList();
            }
        }

        function saveConversations() {
            localStorage.setItem('chatbi-conversations', JSON.stringify(conversations));
        }

        function renderConversationList() {
            const list = el('conversation-list');
            list.innerHTML = '';
            conversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'conversation-item' + (conv.id === currentConversationId ? ' active' : '');
                item.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                    <span class="conversation-title">${escapeHtml(conv.title || 'New Chat')}</span>
                `;
                item.onclick = () => loadConversation(conv.id);
                list.appendChild(item);
            });
        }

        function startNewChat() {
            currentConversationId = null;
            el('welcome-screen').style.display = 'flex';
            el('messages').style.display = 'none';
            el('messages').innerHTML = '';
            el('user-input').value = '';
            autoResize(el('user-input'));
            renderConversationList();
            closeSidebar();
        }

        function loadConversation(id) {
            const conv = conversations.find(c => c.id === id);
            if (!conv) return;
            currentConversationId = id;
            el('welcome-screen').style.display = 'none';
            el('messages').style.display = 'flex';
            el('messages').innerHTML = '';
            conv.messages.forEach(msg => {
                if (msg.role === 'user') {
                    appendUserMessage(msg.content, false);
                } else {
                    appendBotMessage(msg.content, msg.code || '', msg.timing || '', msg.error || '', false);
                }
            });
            renderConversationList();
            closeSidebar();
            setTimeout(scrollToBottom, 50);
        }

        function getOrCreateConversation() {
            if (!currentConversationId) {
                currentConversationId = 'conv_' + Date.now();
                conversations.unshift({
                    id: currentConversationId,
                    title: 'New Chat',
                    messages: [],
                    created: Date.now()
                });
            }
            return conversations.find(c => c.id === currentConversationId);
        }

        // ── Messages ──
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function simpleHighlight(code) {
            // Very light syntax highlighting for Python/pandas
            let highlighted = escapeHtml(code);
            // Comments
            highlighted = highlighted.replace(/(# .*)/g, '<span class="cmt">$1</span>');
            // Strings
            highlighted = highlighted.replace(/(&quot;.*?&quot;)/g, '<span class="str">$1</span>');
            highlighted = highlighted.replace(/(&#x27;.*?&#x27;)/g, '<span class="str">$1</span>');
            // Keywords
            const keywords = ['import', 'from', 'as', 'def', 'return', 'if', 'else', 'elif', 'for', 'in', 'while', 'try', 'except', 'with', 'and', 'or', 'not', 'is', 'None', 'True', 'False'];
            keywords.forEach(kw => {
                const regex = new RegExp(`\\b(${kw})\\b`, 'g');
                highlighted = highlighted.replace(regex, '<span class="kw">$1</span>');
            });
            // Numbers
            highlighted = highlighted.replace(/\b(\d+\.?\d*)\b/g, '<span class="num">$1</span>');
            // Common pandas/numpy functions
            const funcs = ['pd', 'np', 'plt', 'sns', 'read_csv', 'read_excel', 'DataFrame', 'Series', 'groupby', 'agg', 'sum', 'mean', 'count', 'sort_values', 'head', 'tail', 'describe', 'info', 'merge', 'concat', 'pivot_table', 'value_counts', 'plot', 'show', 'to_csv', 'to_excel', 'fillna', 'dropna', 'rename', 'apply', 'map', 'filter', 'query'];
            funcs.forEach(fn => {
                const regex = new RegExp(`\\b(${fn})\\b`, 'g');
                highlighted = highlighted.replace(regex, '<span class="fn">$1</span>');
            });
            return highlighted;
        }

        function appendUserMessage(text, save = true) {
            el('welcome-screen').style.display = 'none';
            el('messages').style.display = 'flex';

            const conv = getOrCreateConversation();
            const msgEl = document.createElement('div');
            msgEl.className = 'message-row user fade-in';
            msgEl.innerHTML = `
                <div class="message-container">
                    <div class="message-content">${escapeHtml(text).replace(/\n/g, '<br>')}</div>
                    <div class="avatar user">U</div>
                </div>
            `;
            el('messages').appendChild(msgEl);

            if (save) {
                conv.messages.push({ role: 'user', content: text });
                if (conv.title === 'New Chat') {
                    conv.title = text.length > 30 ? text.substring(0, 30) + '...' : text;
                }
                saveConversations();
                renderConversationList();
            }
            scrollToBottom();
        }

        function appendTypingIndicator() {
            const msgEl = document.createElement('div');
            msgEl.className = 'message-row bot fade-in';
            msgEl.id = 'typing-indicator';
            msgEl.innerHTML = `
                <div class="message-container">
                    <div class="avatar bot">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="3" y1="9" x2="21" y2="9"></line>
                            <line x1="9" y1="21" x2="9" y2="9"></line>
                        </svg>
                    </div>
                    <div class="message-content">
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            `;
            el('messages').appendChild(msgEl);
            scrollToBottom();
        }

        function removeTypingIndicator() {
            const indicator = el('typing-indicator');
            if (indicator) indicator.remove();
        }

        function appendBotMessage(answer, code, timing, error, save = true) {
            const msgEl = document.createElement('div');
            msgEl.className = 'message-row bot fade-in';

            let html = `<div class="message-container">`;
            html += `<div class="avatar bot">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="3" y1="9" x2="21" y2="9"></line>
                    <line x1="9" y1="21" x2="9" y2="9"></line>
                </svg>
            </div>`;
            html += `<div class="message-content">`;

            if (error) {
                html += `<div class="error-banner">${escapeHtml(error)}</div>`;
            }

            if (answer) {
                html += formatAnswer(answer);
            }

            if (code) {
                html += renderCodeBlock(code);
            }

            if (timing) {
                html += `<div class="timing-info">${escapeHtml(timing)}</div>`;
            }

            html += `</div></div>`;
            msgEl.innerHTML = html;

            el('messages').appendChild(msgEl);

            if (save) {
                const conv = getOrCreateConversation();
                conv.messages.push({ role: 'bot', content: answer, code, timing, error });
                saveConversations();
                renderConversationList();
            }
            scrollToBottom();
        }

        function formatAnswer(text) {
            // Convert markdown-like formatting to HTML
            let html = escapeHtml(text);

            // Bold
            html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            // Inline code
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

            // Line breaks
            html = html.replace(/\n\n/g, '</p><p>');
            html = html.replace(/\n/g, '<br>');

            // Wrap in paragraph if not already
            if (!html.startsWith('<')) {
                html = '<p>' + html + '</p>';
            }

            return html;
        }

        function renderCodeBlock(code) {
            const highlighted = simpleHighlight(code);
            return `
                <div class="code-block-wrapper">
                    <div class="code-block-header">
                        <span class="code-lang-tag">python</span>
                        <div class="code-actions">
                            <button class="code-action-btn" onclick="toggleCodeCollapse(this)" title="Collapse/expand">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="18 15 12 9 6 15"></polyline>
                                </svg>
                            </button>
                            <button class="code-action-btn" onclick="copyCode(this)" title="Copy code">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                                <span class="copy-label">Copy</span>
                            </button>
                        </div>
                    </div>
                    <pre><code>${highlighted}</code></pre>
                </div>
            `;
        }

        function toggleCodeCollapse(btn) {
            const wrapper = btn.closest('.code-block-wrapper');
            const pre = wrapper.querySelector('pre');
            const isCollapsed = pre.style.display === 'none';
            pre.style.display = isCollapsed ? 'block' : 'none';
            btn.querySelector('svg').style.transform = isCollapsed ? 'rotate(0deg)' : 'rotate(180deg)';
        }

        function copyCode(btn) {
            const wrapper = btn.closest('.code-block-wrapper');
            const code = wrapper.querySelector('pre code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                const label = btn.querySelector('.copy-label');
                if (label) label.textContent = 'Copied!';
                showToast('Code copied to clipboard');
                setTimeout(() => { if (label) label.textContent = 'Copy'; }, 2000);
            });
        }

        // ── Input ──
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function sendSuggestion(btn) {
            const title = btn.querySelector('.suggestion-chip-title').textContent;
            const desc = btn.querySelector('.suggestion-chip-desc').textContent.replace(/"/g, '');
            el('user-input').value = desc;
            autoResize(el('user-input'));
            sendMessage();
        }

        async function sendMessage() {
            const input = el('user-input');
            const text = input.value.trim();
            if (!text || isWaiting) return;

            isWaiting = true;
            input.value = '';
            autoResize(input);
            el('send-btn').disabled = true;

            appendUserMessage(text);
            appendTypingIndicator();

            const startTime = performance.now();

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: text })
                });

                const data = await response.json();
                removeTypingIndicator();

                const elapsed = Math.round(performance.now() - startTime);
                const timingStr = data.total_time_ms
                    ? `${data.total_time_ms}ms`
                    : `${elapsed}ms`;

                if (data.success) {
                    appendBotMessage(
                        data.answer || '',
                        data.code || '',
                        timingStr,
                        ''
                    );
                } else {
                    appendBotMessage(
                        data.answer || 'Something went wrong. Please try again.',
                        data.code || '',
                        timingStr,
                        data.error || ''
                    );
                }
            } catch (err) {
                removeTypingIndicator();
                appendBotMessage(
                    'Unable to reach the server. Please make sure the backend is running.',
                    '',
                    '',
                    err.message
                );
            } finally {
                isWaiting = false;
                el('send-btn').disabled = false;
                el('user-input').focus();
            }
        }

        // ── Scroll ──
        function scrollToBottom() {
            const chatArea = el('chat-area');
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        el('chat-area').addEventListener('scroll', () => {
            const chatArea = el('chat-area');
            const isAtBottom = chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 60;
            el('scroll-to-bottom').classList.toggle('visible', !isAtBottom);
        });

        // ── Sidebar Mobile ──
        function toggleSidebar() {
            el('sidebar').classList.toggle('open');
        }

        function closeSidebar() {
            el('sidebar').classList.remove('open');
        }

        // ── Toast ──
        function showToast(message) {
            const toast = el('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2500);
        }

        // ── Health Check ──
        async function checkHealth() {
            try {
                const res = await fetch('/api/health');
                const data = await res.json();
                const dot = el('status-dot');
                if (data.status === 'ok') {
                    dot.classList.remove('offline');
                    el('model-name').textContent = data.model || MODEL;
                } else {
                    dot.classList.add('offline');
                }
            } catch {
                el('status-dot').classList.add('offline');
            }
        }

        // ── Init ──
        document.addEventListener('DOMContentLoaded', () => {
            initTheme();
            loadConversations();
            checkHealth();
            setInterval(checkHealth, 30000);
            el('user-input').focus();
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main chat UI."""
    return render_template_string(INDEX_HTML, model=OLLAMA_MODEL)


@app.route("/api/health")
def health():
    """Health check endpoint."""
    agent = get_agent()
    model_name = OLLAMA_MODEL
    status_ok = agent is not None

    # Try to verify the agent is actually working
    if agent is not None:
        try:
            # Quick sanity check - just verify attributes exist
            hasattr(agent, "answer")
        except Exception as e:
            logger.warning("Agent health check warning: %s", e)

    return jsonify({"status": "ok" if status_ok else "degraded", "model": model_name})


@app.route("/api/ask", methods=["POST"])
def ask():
    """Ask the agent a question and return the response."""
    start_time = time.time()
    data = request.get_json(force=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "success": False,
            "answer": "",
            "code": "",
            "error": "Question cannot be empty.",
            "total_time_ms": 0,
        }), 400

    agent = get_agent()
    if agent is None:
        total_ms = round((time.time() - start_time) * 1000)
        return jsonify({
            "success": False,
            "answer": "",
            "code": "",
            "error": "ChatBIAgent is not available. Please check your installation.",
            "total_time_ms": total_ms,
        }), 503

    try:
        logger.info("Question: %s", question)
        result = agent.answer(question)
        total_ms = round((time.time() - start_time) * 1000)

        # AgentResult is a dataclass — use attributes directly
        response = {
            "success": result.success,
            "answer": result.answer,
            "code": result.code,
            "error": result.error,
            "total_time_ms": total_ms,
            "timings": [{"stage": t.stage, "elapsed_ms": t.elapsed_ms} for t in result.timings],
            "retries": result.retries,
        }

        logger.info("Answer received in %dms (success=%s)", total_ms, response["success"])
        return jsonify(response)

    except Exception as e:
        total_ms = round((time.time() - start_time) * 1000)
        logger.error("Error processing question: %s", e, exc_info=True)
        return jsonify({
            "success": False,
            "answer": "An internal error occurred while processing your request.",
            "code": "",
            "error": str(e),
            "total_time_ms": total_ms,
        }), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  ChatBI Agent - Web GUI")
    logger.info("  Model: %s", OLLAMA_MODEL)
    logger.info("  Ollama: %s", OLLAMA_URL)
    logger.info("  Listening on http://%s:%d", HOST, PORT)
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 60)

    app.run(host=HOST, port=PORT, debug=False, threaded=True)
