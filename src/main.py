"""
Chess Oracle - Árbitro Literário de Xadrez

Entry point do MVP que orquestra todos os componentes:
- RAG Engine para busca em literatura clássica
- Chess Analyzer para avaliação com Maia Engine
- LLM (Groq/Llama-3) para síntese e explicações

Este é o "cérebro" que combina sabedoria clássica com análise moderna.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

from rag_engine import ChessRAGEngine
from chess_logic import ChessAnalyzer
from prompt_template import ChessOraclePrompts

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Biblioteca 'groq' não instalada. Instale com: pip install groq")


class ChessOracle:
    """
    Orquestrador principal do Chess Oracle.
    
    Combina:
    - Busca semântica em livros clássicos (RAG)
    - Análise técnica com Maia Chess (UCI)
    - Síntese em linguagem natural (LLM)
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        engine_path: Optional[str] = None,
        llm_model: str = "llama-3.1-70b-versatile"
    ):
        """
        Inicializa o Chess Oracle.
        
        Args:
            groq_api_key: API key do Groq (usa .env se None)
            engine_path: Caminho para Maia engine (usa .env se None)
            llm_model: Nome do modelo LLM no Groq
        """
        # Carrega variáveis de ambiente
        load_dotenv()
        
        # Inicializa componentes
        self.rag_engine = ChessRAGEngine()
        self.chess_analyzer = ChessAnalyzer(engine_path=engine_path)
        self.prompts = ChessOraclePrompts()
        self.llm_model = llm_model
        
        # Inicializa cliente Groq
        if GROQ_AVAILABLE:
            api_key = groq_api_key or os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
                self.llm_enabled = True
            else:
                print("⚠️  GROQ_API_KEY não encontrada. Modo LLM desabilitado.")
                self.llm_enabled = False
        else:
            self.llm_enabled = False
    
    def analyze(
        self,
        fen: str,
        user_query: str,
        engine_depth: int = 20,
        rag_k: int = 3,
        engine_time: float = 1.0
    ) -> Dict:
        """
        Método principal de análise do Chess Oracle.
        
        Fluxo de execução:
        1. Consulta ChromaDB por contexto histórico relevante
        2. Consulta Maia Engine para avaliação técnica
        3. Envia para LLM com prompt de comparação
        4. Retorna resposta sintetizada
        
        Args:
            fen: Posição FEN a analisar
            user_query: Pergunta do usuário
            engine_depth: Profundidade de análise da engine (padrão: 20)
            rag_k: Número de contextos a buscar (padrão: 3)
            engine_time: Tempo máximo de análise em segundos (padrão: 1.0)
            
        Returns:
            Dict contendo:
                - 'success': bool indicando sucesso
                - 'response': Resposta do LLM
                - 'technical_analysis': Análise da engine
                - 'historical_context': Contexto dos livros
                - 'error': Mensagem de erro (se aplicável)
        """
        result = {
            'success': False,
            'response': None,
            'technical_analysis': None,
            'historical_context': None,
            'error': None
        }
        
        try:
            # 1. Valida posição FEN
            is_valid, error = self.chess_analyzer.validate_fen(fen)
            if not is_valid:
                result['error'] = f"FEN inválido: {error}"
                return result
            
            # 2. Busca contexto histórico no RAG
            print("🔍 Buscando contexto histórico...")
            historical_context = self.rag_engine.search_context(
                query=user_query,
                k=rag_k
            )
            result['historical_context'] = historical_context
            
            # 3. Analisa posição com Maia Engine
            print("♟️  Analisando posição com Maia Chess...")
            try:
                technical_analysis = self.chess_analyzer.analyze_position(
                    fen=fen,
                    depth=engine_depth,
                    time_limit=engine_time
                )
                result['technical_analysis'] = technical_analysis
            except FileNotFoundError as e:
                result['error'] = str(e)
                print(f"⚠️  {e}")
                # Continua sem análise da engine
                technical_analysis = {
                    'position_fen': fen,
                    'best_move': None,
                    'evaluation': None,
                    'is_mate': False,
                    'mate_in': None
                }
            
            # 4. Gera prompt para LLM
            if historical_context:
                prompt = self.prompts.format_comparison_prompt(
                    historical_context=historical_context,
                    position_fen=fen,
                    evaluation=technical_analysis.get('evaluation'),
                    best_move=technical_analysis.get('best_move'),
                    user_query=user_query,
                    is_mate=technical_analysis.get('is_mate', False),
                    mate_in=technical_analysis.get('mate_in')
                )
            else:
                prompt = self.prompts.format_no_context_prompt(
                    position_fen=fen,
                    evaluation=technical_analysis.get('evaluation'),
                    best_move=technical_analysis.get('best_move'),
                    user_query=user_query,
                    is_mate=technical_analysis.get('is_mate', False),
                    mate_in=technical_analysis.get('mate_in')
                )
            
            # 5. Consulta LLM
            if self.llm_enabled:
                print("🤖 Consultando LLM...")
                llm_response = self._query_llm(prompt)
                result['response'] = llm_response
                result['success'] = True
            else:
                result['response'] = self._generate_fallback_response(
                    technical_analysis,
                    historical_context
                )
                result['success'] = True
                result['error'] = "LLM não disponível. Resposta básica gerada."
            
        except Exception as e:
            result['error'] = f"Erro durante análise: {str(e)}"
            print(f"❌ {result['error']}")
        
        finally:
            # Fecha engine se foi aberta
            self.chess_analyzer.close_engine()
        
        return result
    
    def _query_llm(self, prompt: str) -> str:
        """
        Consulta o LLM via Groq API.
        
        Args:
            prompt: Prompt formatado
            
        Returns:
            Resposta do LLM
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.prompts.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.llm_model,
                temperature=0.7,
                max_tokens=2000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Erro ao consultar LLM: {str(e)}"
    
    def _generate_fallback_response(
        self,
        technical_analysis: Dict,
        historical_context: list
    ) -> str:
        """
        Gera resposta básica quando LLM não está disponível.
        
        Args:
            technical_analysis: Análise da engine
            historical_context: Contexto histórico
            
        Returns:
            Resposta formatada em texto
        """
        response_parts = ["=== Análise Básica ===\n"]
        
        # Análise técnica
        if technical_analysis:
            response_parts.append("## Análise da Engine:")
            eval_val = technical_analysis.get('evaluation')
            if eval_val is not None:
                response_parts.append(f"- Avaliação: {eval_val/100:+.2f}")
            if technical_analysis.get('best_move'):
                response_parts.append(f"- Melhor movimento: {technical_analysis['best_move']}")
            if technical_analysis.get('is_mate'):
                response_parts.append(f"- Mate em {technical_analysis.get('mate_in')} movimentos")
        
        # Contexto histórico
        if historical_context:
            response_parts.append("\n## Contexto Histórico:")
            for i, ctx in enumerate(historical_context, 1):
                source = ctx['source'].split('/')[-1]
                response_parts.append(f"\n{i}. Fonte: {source} (p.{ctx['page']})")
                response_parts.append(f"   {ctx['text'][:150]}...")
        
        response_parts.append("\n⚠️  Para análise completa, configure GROQ_API_KEY no arquivo .env")
        
        return "\n".join(response_parts)
    
    def load_books(self, data_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Carrega livros de xadrez no banco vetorial.
        
        Args:
            data_dir: Diretório contendo PDFs (padrão: ./data)
            
        Returns:
            Dict mapeando arquivo para número de chunks
        """
        print("📚 Carregando livros de xadrez...")
        return self.rag_engine.ingest_directory(data_dir)
    
    def get_stats(self) -> Dict:
        """
        Retorna estatísticas do sistema.
        
        Returns:
            Dict com estatísticas dos componentes
        """
        rag_stats = self.rag_engine.get_collection_stats()
        return {
            'rag': rag_stats,
            'llm_enabled': self.llm_enabled,
            'llm_model': self.llm_model if self.llm_enabled else None
        }


# Exemplo de uso do MVP
if __name__ == "__main__":
    print("=" * 60)
    print("♟️  Chess Oracle - Árbitro Literário de Xadrez")
    print("=" * 60)
    
    # Inicializa o Oracle
    oracle = ChessOracle()
    
    # Exibe estatísticas
    stats = oracle.get_stats()
    print(f"\n📊 Estatísticas do Sistema:")
    print(f"   - Base vetorial: {stats['rag']['total_chunks']} chunks")
    print(f"   - Modelo embedding: {stats['rag']['embedding_model']}")
    print(f"   - LLM ativo: {'✓' if stats['llm_enabled'] else '✗'}")
    if stats['llm_model']:
        print(f"   - Modelo LLM: {stats['llm_model']}")
    
    # Se não há livros, sugere carregar
    if stats['rag']['total_chunks'] == 0:
        print("\n📚 Nenhum livro carregado ainda.")
        print("   Para carregar livros: oracle.load_books()")
        print("   Coloque arquivos PDF em ./data/ primeiro")
    
    # Exemplo de análise
    print("\n" + "=" * 60)
    print("Exemplo de Análise")
    print("=" * 60)
    
    # Posição da Abertura Ruy Lopez
    ruy_lopez_fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    query = "Por que o movimento Bb5 (Ruy Lopez) é considerado tão forte?"
    
    print(f"\nPosição FEN: {ruy_lopez_fen}")
    print(f"Consulta: {query}\n")
    
    result = oracle.analyze(
        fen=ruy_lopez_fen,
        user_query=query,
        engine_depth=15  # Reduzido para exemplo rápido
    )
    
    if result['success']:
        print("✓ Análise concluída!\n")
        print(result['response'])
    else:
        print(f"✗ Erro na análise: {result['error']}")
    
    print("\n" + "=" * 60)
    print("Para usar o Chess Oracle:")
    print("1. Instale dependências: pip install -r requirements.txt")
    print("2. Configure GROQ_API_KEY no arquivo .env")
    print("3. Baixe Maia Chess e coloque em ./engine/")
    print("4. Adicione PDFs de livros em ./data/")
    print("5. Execute: python src/main.py")
    print("=" * 60)
