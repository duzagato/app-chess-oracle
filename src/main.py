"""
Chess Oracle - Entry Point Principal

Orquestra todos os componentes do sistema: RAG, Engine de Xadrez e LLM local (Ollama)
para fornecer análises que combinam literatura clássica com avaliações modernas.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

from chess_logic import ChessAnalyzer
from rag_engine import ChessRAGEngine
from llm_engine import OllamaLLM
from prompt_template import ChessOraclePrompts


class ChessOracle:
    """
    Orquestrador principal do Chess Oracle.
    
    Integra RAG (busca em livros), análise de engine e LLM local para fornecer
    análises completas que combinam sabedoria clássica e precisão moderna.
    """
    
    def __init__(
        self,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
        maia_engine_path: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        auto_initialize: bool = True
    ):
        """
        Inicializa o Chess Oracle.
        
        Args:
            ollama_host: URL do servidor Ollama
            ollama_model: Modelo Ollama a usar
            maia_engine_path: Caminho para engine Maia
            vector_db_path: Diretório do banco vetorial
            auto_initialize: Se True, inicializa componentes automaticamente
        """
        # Carrega variáveis de ambiente
        load_dotenv()
        
        # Configurações
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_model = ollama_model or os.getenv('OLLAMA_MODEL', 'llama3')
        self.maia_engine_path = maia_engine_path or os.getenv('MAIA_ENGINE_PATH', './engine/maia-1900.elf')
        self.vector_db_path = vector_db_path or os.getenv('CHROMA_PERSIST_DIRECTORY', './vector_db')
        
        # Componentes
        self.chess_analyzer: Optional[ChessAnalyzer] = None
        self.rag_engine: Optional[ChessRAGEngine] = None
        self.llm: Optional[OllamaLLM] = None
        self.prompts = ChessOraclePrompts()
        
        # Estado
        self.components_status = {
            'chess_engine': False,
            'rag': False,
            'ollama': False
        }
        
        if auto_initialize:
            self.initialize()
    
    def initialize(self) -> Dict[str, bool]:
        """
        Inicializa todos os componentes do sistema.
        
        Returns:
            Dicionário com status de cada componente
        """
        print("\n" + "="*60)
        print("Inicializando Chess Oracle")
        print("="*60 + "\n")
        
        # 1. Inicializa analisador de xadrez
        print("1. Analisador de Xadrez...")
        try:
            self.chess_analyzer = ChessAnalyzer(self.maia_engine_path)
            engine_available = self.chess_analyzer.initialize_engine()
            self.components_status['chess_engine'] = engine_available
            
            if engine_available:
                print("   ✓ Engine Maia inicializada\n")
            else:
                print("   ⚠️ Engine Maia não disponível (apenas validação de posições)\n")
        except Exception as e:
            print(f"   ⚠️ Erro ao inicializar analisador: {e}\n")
            self.components_status['chess_engine'] = False
        
        # 2. Inicializa RAG engine
        print("2. RAG Engine (Busca em Livros)...")
        try:
            self.rag_engine = ChessRAGEngine(
                persist_directory=self.vector_db_path
            )
            stats = self.rag_engine.get_collection_stats()
            self.components_status['rag'] = True
            print(f"   ✓ RAG inicializado: {stats['total_documents']} documentos\n")
        except Exception as e:
            print(f"   ⚠️ Erro ao inicializar RAG: {e}\n")
            self.components_status['rag'] = False
        
        # 3. Inicializa Ollama LLM
        print("3. Ollama LLM (Gerador de Linguagem Natural)...")
        try:
            self.llm = OllamaLLM(
                host=self.ollama_host,
                model=self.ollama_model
            )
            
            health = self.llm.get_health_check()
            
            if health['running']:
                if health['model_available']:
                    self.components_status['ollama'] = True
                    print(f"   ✓ Ollama rodando com modelo '{self.ollama_model}'\n")
                else:
                    print(f"   ⚠️ Ollama rodando mas modelo '{self.ollama_model}' não encontrado")
                    print(f"   Execute: ollama pull {self.ollama_model}\n")
                    self.components_status['ollama'] = False
            else:
                print(f"   ⚠️ Servidor Ollama não está rodando")
                print(f"   Execute: ollama serve\n")
                self.components_status['ollama'] = False
        except Exception as e:
            print(f"   ⚠️ Erro ao inicializar Ollama: {e}\n")
            self.components_status['ollama'] = False
        
        # Resumo
        print("="*60)
        print("Status dos Componentes:")
        print(f"  Chess Engine: {'✓' if self.components_status['chess_engine'] else '✗'}")
        print(f"  RAG (Livros): {'✓' if self.components_status['rag'] else '✗'}")
        print(f"  Ollama LLM:   {'✓' if self.components_status['ollama'] else '✗'}")
        print("="*60 + "\n")
        
        return self.components_status
    
    def analyze(
        self,
        fen: str,
        user_query: str,
        use_rag: bool = True,
        use_engine: bool = True,
        rag_k: int = 3
    ) -> Dict:
        """
        Realiza análise completa de uma posição.
        
        Args:
            fen: Posição em notação FEN
            user_query: Pergunta ou contexto do usuário
            use_rag: Se deve buscar contexto em livros
            use_engine: Se deve usar engine para análise
            rag_k: Número de documentos a buscar no RAG
        
        Returns:
            Dicionário com análise completa:
            - 'fen': FEN da posição
            - 'query': Query original
            - 'rag_context': Contexto dos livros (se disponível)
            - 'engine_analysis': Análise da engine (se disponível)
            - 'oracle_response': Resposta sintetizada do LLM
            - 'components_used': Lista de componentes utilizados
        """
        result = {
            'fen': fen,
            'query': user_query,
            'components_used': [],
            'warnings': []
        }
        
        # 1. Valida FEN
        if self.chess_analyzer:
            valid, message = self.chess_analyzer.validate_fen(fen)
            if not valid:
                result['error'] = message
                return result
        
        # 2. Busca contexto no RAG
        rag_context = None
        if use_rag and self.components_status['rag'] and self.rag_engine:
            try:
                rag_results = self.rag_engine.search_context(user_query, k=rag_k)
                
                if rag_results:
                    # Formata contexto dos livros
                    context_parts = []
                    for i, res in enumerate(rag_results, 1):
                        source = res['metadata']['filename']
                        page = res['metadata']['page']
                        text = res['text']
                        context_parts.append(
                            f"[Fonte {i}: {source}, p.{page}]\n{text}\n"
                        )
                    
                    rag_context = "\n".join(context_parts)
                    result['rag_context'] = rag_context
                    result['components_used'].append('rag')
            except Exception as e:
                result['warnings'].append(f"Erro no RAG: {e}")
        
        # 3. Analisa com engine
        engine_analysis = None
        if use_engine and self.components_status['chess_engine'] and self.chess_analyzer:
            try:
                engine_analysis = self.chess_analyzer.analyze_position(fen)
                result['engine_analysis'] = engine_analysis
                result['components_used'].append('engine')
            except Exception as e:
                result['warnings'].append(f"Erro na engine: {e}")
        
        # 4. Gera resposta com LLM
        if self.components_status['ollama'] and self.llm:
            try:
                # Prepara prompt
                prompt = self.prompts.get_analysis_prompt(
                    fen=fen,
                    user_query=user_query,
                    book_context=rag_context,
                    engine_evaluation=engine_analysis
                )
                
                system_prompt = self.prompts.get_system_prompt()
                
                # Gera resposta
                response = self.llm.generate(
                    prompt=prompt,
                    system=system_prompt,
                    temperature=0.7
                )
                
                if 'error' not in response:
                    result['oracle_response'] = response['response']
                    result['components_used'].append('ollama')
                else:
                    result['warnings'].append(f"Erro no Ollama: {response['error']}")
                    result['oracle_response'] = self._generate_fallback_response(
                        fen, user_query, rag_context, engine_analysis
                    )
            except Exception as e:
                result['warnings'].append(f"Erro ao gerar resposta: {e}")
                result['oracle_response'] = self._generate_fallback_response(
                    fen, user_query, rag_context, engine_analysis
                )
        else:
            result['warnings'].append("Ollama não disponível - resposta fallback")
            result['oracle_response'] = self._generate_fallback_response(
                fen, user_query, rag_context, engine_analysis
            )
        
        return result
    
    def _generate_fallback_response(
        self,
        fen: str,
        query: str,
        rag_context: Optional[str],
        engine_analysis: Optional[Dict]
    ) -> str:
        """
        Gera resposta básica quando LLM não está disponível.
        
        Args:
            fen: Posição FEN
            query: Query do usuário
            rag_context: Contexto do RAG
            engine_analysis: Análise da engine
        
        Returns:
            String com resposta formatada
        """
        parts = [
            "# Análise da Posição\n",
            f"**FEN:** {fen}",
            f"**Pergunta:** {query}\n"
        ]
        
        if rag_context:
            parts.append("## Contexto dos Livros")
            parts.append(rag_context)
        
        if engine_analysis and engine_analysis.get('engine_available'):
            parts.append("## Análise da Engine")
            parts.append(f"Avaliação: {engine_analysis.get('evaluation', 'N/A')}")
            parts.append(f"Melhor lance: {engine_analysis.get('best_move_san', 'N/A')}")
        
        parts.append("\n⚠️ Resposta gerada sem LLM (Ollama não disponível)")
        
        return "\n".join(parts)
    
    def close(self):
        """Fecha todos os recursos."""
        if self.chess_analyzer:
            self.chess_analyzer.close_engine()


def main():
    """Função principal - exemplo de uso."""
    print("\n" + "="*70)
    print(" "*20 + "♟️  CHESS ORACLE ♟️")
    print("="*70)
    print("\nÁrbitro Literário: Literatura Clássica + IA Moderna\n")
    
    # Inicializa o Oracle
    oracle = ChessOracle()
    
    # Verifica se pelo menos um componente está funcionando
    if not any(oracle.components_status.values()):
        print("⚠️ Nenhum componente disponível. Verifique a configuração.")
        return
    
    # Exemplo de análise
    print("\n" + "="*70)
    print("EXEMPLO DE ANÁLISE")
    print("="*70 + "\n")
    
    # Posição: Defesa Siciliana após 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4
    fen = "rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4"
    query = "Esta é uma posição típica da Defesa Siciliana. Quais são os planos para as pretas?"
    
    print(f"Posição: Defesa Siciliana (FEN)")
    print(f"Pergunta: {query}\n")
    
    result = oracle.analyze(fen, query)
    
    # Mostra resultado
    if 'error' in result:
        print(f"❌ Erro: {result['error']}")
    else:
        print("Componentes utilizados:", ", ".join(result['components_used']) or "nenhum")
        
        if result.get('warnings'):
            print("\nAvisos:")
            for warning in result['warnings']:
                print(f"  ⚠️ {warning}")
        
        print("\n" + "-"*70)
        print("RESPOSTA DO CHESS ORACLE:")
        print("-"*70 + "\n")
        print(result.get('oracle_response', 'Nenhuma resposta gerada'))
    
    # Fecha recursos
    oracle.close()
    
    print("\n" + "="*70)
    print("Análise concluída!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
