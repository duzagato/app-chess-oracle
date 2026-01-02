"""
Chess Oracle - Árbitro Literário de Xadrez

Package que combina literatura clássica de xadrez com análise moderna de engines,
executando 100% localmente usando Ollama.
"""

__version__ = "0.1.0"
__author__ = "Chess Oracle Team"

from .main import ChessOracle
from .chess_logic import ChessAnalyzer
from .rag_engine import ChessRAGEngine
from .llm_engine import OllamaLLM
from .prompt_template import ChessOraclePrompts

__all__ = [
    'ChessOracle',
    'ChessAnalyzer',
    'ChessRAGEngine',
    'OllamaLLM',
    'ChessOraclePrompts'
]
