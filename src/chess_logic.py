"""
Módulo de análise de xadrez integrado com python-chess e Maia Chess Engine.

Este módulo fornece funcionalidades para:
- Validar posições FEN e PGN
- Analisar posições usando a engine Maia Chess via protocolo UCI
- Extrair avaliações e melhores movimentos
"""

import chess
import chess.engine
import os
from typing import Dict, Optional, List, Tuple
from pathlib import Path


class ChessAnalyzer:
    """
    Classe para análise de posições de xadrez usando python-chess e Maia Chess.
    
    A Maia Chess é uma engine que pensa como humanos, ideal para detectar
    erros naturais e fornecer análises compreensíveis.
    """
    
    def __init__(self, engine_path: Optional[str] = None):
        """
        Inicializa o analisador de xadrez.
        
        Args:
            engine_path: Caminho para o binário da Maia Chess. 
                        Se None, usa MAIA_ENGINE_PATH do .env
        """
        self.engine_path = engine_path or os.getenv('MAIA_ENGINE_PATH', './engine/maia')
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.board: Optional[chess.Board] = None
        
    def _initialize_engine(self) -> None:
        """Inicializa a conexão com a engine Maia via protocolo UCI."""
        if self.engine is None:
            engine_path = Path(self.engine_path)
            if not engine_path.exists():
                raise FileNotFoundError(
                    f"Engine Maia não encontrada em: {self.engine_path}. "
                    "Por favor, baixe o binário da Maia Chess e coloque no diretório correto."
                )
            self.engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    
    def close_engine(self) -> None:
        """Fecha a conexão com a engine."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None
    
    def validate_fen(self, fen: str) -> Tuple[bool, Optional[str]]:
        """
        Valida uma posição FEN.
        
        Args:
            fen: String FEN representando uma posição
            
        Returns:
            Tuple (válido: bool, erro: Optional[str])
        """
        try:
            chess.Board(fen)
            return True, None
        except ValueError as e:
            return False, str(e)
    
    def validate_pgn(self, pgn: str) -> Tuple[bool, Optional[str]]:
        """
        Valida uma string PGN.
        
        Args:
            pgn: String PGN representando uma partida
            
        Returns:
            Tuple (válido: bool, erro: Optional[str])
        """
        try:
            import io
            import chess.pgn
            
            pgn_io = io.StringIO(pgn)
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                return False, "PGN inválido ou vazio"
            return True, None
        except Exception as e:
            return False, str(e)
    
    def load_position(self, fen: str) -> bool:
        """
        Carrega uma posição FEN no tabuleiro interno.
        
        Args:
            fen: String FEN da posição
            
        Returns:
            True se carregada com sucesso, False caso contrário
        """
        is_valid, error = self.validate_fen(fen)
        if not is_valid:
            raise ValueError(f"FEN inválido: {error}")
        
        self.board = chess.Board(fen)
        return True
    
    def analyze_position(
        self, 
        fen: str, 
        depth: int = 20,
        time_limit: float = 1.0
    ) -> Dict:
        """
        Analisa uma posição usando a Maia Chess Engine.
        
        Args:
            fen: String FEN da posição a analisar
            depth: Profundidade de análise (padrão: 20)
            time_limit: Tempo máximo de análise em segundos (padrão: 1.0)
            
        Returns:
            Dict contendo:
                - 'evaluation': Avaliação em centipawns (perspectiva das brancas)
                - 'best_move': Melhor movimento sugerido (notação UCI)
                - 'is_mate': Se há mate forçado
                - 'mate_in': Número de movimentos até o mate (se aplicável)
                - 'position_fen': FEN da posição analisada
        """
        self.load_position(fen)
        self._initialize_engine()
        
        # Analisa a posição
        limit = chess.engine.Limit(depth=depth, time=time_limit)
        info = self.engine.analyse(self.board, limit)
        
        # Extrai informações da análise
        score = info.get('score')
        best_move = info.get('pv', [None])[0]  # Principal Variation
        
        result = {
            'position_fen': fen,
            'best_move': str(best_move) if best_move else None,
            'is_mate': False,
            'mate_in': None,
            'evaluation': None
        }
        
        # Processa a pontuação
        if score:
            # Converte para perspectiva das brancas
            white_score = score.white()
            
            if white_score.is_mate():
                result['is_mate'] = True
                result['mate_in'] = white_score.mate()
            else:
                result['evaluation'] = white_score.score()  # Em centipawns
        
        return result
    
    def get_legal_moves(self, fen: Optional[str] = None) -> List[str]:
        """
        Retorna todos os movimentos legais para a posição atual.
        
        Args:
            fen: FEN da posição (usa posição carregada se None)
            
        Returns:
            Lista de movimentos em notação UCI
        """
        if fen:
            self.load_position(fen)
        
        if self.board is None:
            raise ValueError("Nenhuma posição carregada")
        
        return [move.uci() for move in self.board.legal_moves]
    
    def is_game_over(self, fen: Optional[str] = None) -> bool:
        """
        Verifica se o jogo terminou (mate, afogamento, etc).
        
        Args:
            fen: FEN da posição (usa posição carregada se None)
            
        Returns:
            True se o jogo terminou
        """
        if fen:
            self.load_position(fen)
        
        if self.board is None:
            raise ValueError("Nenhuma posição carregada")
        
        return self.board.is_game_over()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - fecha a engine."""
        self.close_engine()


# Exemplo de uso
if __name__ == "__main__":
    # Posição inicial do xadrez
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    analyzer = ChessAnalyzer()
    
    # Valida FEN
    is_valid, error = analyzer.validate_fen(initial_fen)
    print(f"FEN válido: {is_valid}")
    
    if is_valid:
        # Nota: Isso só funcionará se a engine Maia estiver instalada
        try:
            result = analyzer.analyze_position(initial_fen, depth=15)
            print(f"Análise: {result}")
        except FileNotFoundError as e:
            print(f"Engine não encontrada: {e}")
        finally:
            analyzer.close_engine()
