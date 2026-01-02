"""
Módulo de lógica de xadrez e integração com engine Maia Chess.

Este módulo fornece a classe ChessAnalyzer para validar posições de xadrez
e obter análises técnicas usando a engine Maia via protocolo UCI.
"""

import chess
import chess.engine
from typing import Dict, Optional, Tuple
import os


class ChessAnalyzer:
    """
    Analisador de xadrez que integra python-chess e Maia Chess engine.
    
    A classe fornece métodos para:
    - Validar posições FEN/PGN
    - Analisar posições usando engine Maia
    - Obter avaliações numéricas e melhores jogadas
    """
    
    def __init__(self, engine_path: Optional[str] = None):
        """
        Inicializa o analisador de xadrez.
        
        Args:
            engine_path: Caminho para o binário da engine Maia.
                        Se None, usa a variável de ambiente MAIA_ENGINE_PATH.
        """
        self.engine_path = engine_path or os.getenv('MAIA_ENGINE_PATH', './engine/maia-1900.elf')
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.board = chess.Board()
    
    def initialize_engine(self) -> bool:
        """
        Inicializa a engine Maia se o binário estiver disponível.
        
        Returns:
            True se a engine foi inicializada com sucesso, False caso contrário.
        """
        if not os.path.exists(self.engine_path):
            print(f"⚠️ Engine Maia não encontrada em: {self.engine_path}")
            print("   Continuando sem análise de engine...")
            return False
        
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            print(f"✓ Engine Maia inicializada: {self.engine_path}")
            return True
        except Exception as e:
            print(f"⚠️ Erro ao inicializar engine: {e}")
            return False
    
    def close_engine(self):
        """Fecha a conexão com a engine se estiver aberta."""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def validate_fen(self, fen: str) -> Tuple[bool, str]:
        """
        Valida uma string FEN.
        
        Args:
            fen: String no formato FEN (Forsyth-Edwards Notation)
        
        Returns:
            Tupla (válido, mensagem) onde válido é True se o FEN é válido
        """
        try:
            chess.Board(fen)
            return True, "FEN válido"
        except ValueError as e:
            return False, f"FEN inválido: {str(e)}"
    
    def validate_pgn(self, pgn: str) -> Tuple[bool, str]:
        """
        Valida uma sequência PGN de movimentos.
        
        Args:
            pgn: String com movimentos em formato PGN
        
        Returns:
            Tupla (válido, mensagem) com status da validação
        """
        try:
            board = chess.Board()
            moves = pgn.split()
            
            for move in moves:
                # Remove números de movimento e indicadores de resultado
                move = move.replace('.', '').strip()
                if move in ['1-0', '0-1', '1/2-1/2', '*']:
                    continue
                if move and not move.isdigit():
                    board.push_san(move)
            
            return True, f"PGN válido - {len(board.move_stack)} movimentos"
        except Exception as e:
            return False, f"PGN inválido: {str(e)}"
    
    def set_position(self, fen: str) -> bool:
        """
        Define a posição atual do tabuleiro usando FEN.
        
        Args:
            fen: String FEN da posição
        
        Returns:
            True se a posição foi definida com sucesso
        """
        try:
            self.board = chess.Board(fen)
            return True
        except ValueError:
            return False
    
    def analyze_position(self, fen: str, depth: int = 20, time_limit: float = 2.0) -> Dict:
        """
        Analisa uma posição usando a engine Maia.
        
        Args:
            fen: String FEN da posição a analisar
            depth: Profundidade de busca (default: 20)
            time_limit: Tempo máximo de análise em segundos (default: 2.0)
        
        Returns:
            Dicionário com resultados da análise:
            - 'fen': FEN da posição
            - 'evaluation': Avaliação numérica (em centipawns ou mate)
            - 'best_move': Melhor movimento em notação UCI
            - 'best_move_san': Melhor movimento em notação SAN
            - 'engine_available': Se a engine está disponível
            - 'legal_moves': Lista de movimentos legais
        """
        # Valida o FEN
        valid, message = self.validate_fen(fen)
        if not valid:
            return {
                'fen': fen,
                'error': message,
                'engine_available': False
            }
        
        # Define a posição
        self.set_position(fen)
        
        result = {
            'fen': fen,
            'legal_moves': [self.board.san(move) for move in self.board.legal_moves],
            'engine_available': self.engine is not None
        }
        
        # Se a engine não está disponível, retorna apenas informações básicas
        if not self.engine:
            result['evaluation'] = 'Engine não disponível'
            result['best_move'] = None
            result['best_move_san'] = None
            return result
        
        # Analisa a posição com a engine
        try:
            info = self.engine.analyse(
                self.board,
                chess.engine.Limit(depth=depth, time=time_limit)
            )
            
            # Extrai avaliação
            score = info.get('score')
            if score:
                # Converte score relativo ao lado que está jogando
                score = score.white() if self.board.turn == chess.WHITE else score.black()
                
                if score.is_mate():
                    mate_in = score.mate()
                    result['evaluation'] = f"Mate em {abs(mate_in)}" if mate_in else "Mate"
                    result['evaluation_cp'] = 10000 if mate_in > 0 else -10000
                else:
                    cp = score.score()
                    result['evaluation'] = f"{cp / 100:.2f}" if cp else "0.00"
                    result['evaluation_cp'] = cp
            
            # Extrai melhor movimento
            pv = info.get('pv')
            if pv and len(pv) > 0:
                best_move = pv[0]
                result['best_move'] = best_move.uci()
                result['best_move_san'] = self.board.san(best_move)
            
        except Exception as e:
            result['error'] = f"Erro na análise: {str(e)}"
        
        return result
    
    def get_position_info(self, fen: str) -> Dict:
        """
        Obtém informações gerais sobre uma posição.
        
        Args:
            fen: String FEN da posição
        
        Returns:
            Dicionário com informações da posição:
            - 'turn': Quem joga ('white' ou 'black')
            - 'castling': Direitos de roque disponíveis
            - 'en_passant': Casa de en passant se disponível
            - 'halfmove_clock': Contador de meios-lances
            - 'fullmove_number': Número do lance completo
            - 'is_check': Se o rei está em xeque
            - 'is_checkmate': Se é xeque-mate
            - 'is_stalemate': Se é afogamento
            - 'is_game_over': Se o jogo terminou
        """
        if not self.set_position(fen):
            return {'error': 'FEN inválido'}
        
        return {
            'turn': 'white' if self.board.turn == chess.WHITE else 'black',
            'castling': self.board.castling_rights,
            'en_passant': self.board.ep_square,
            'halfmove_clock': self.board.halfmove_clock,
            'fullmove_number': self.board.fullmove_number,
            'is_check': self.board.is_check(),
            'is_checkmate': self.board.is_checkmate(),
            'is_stalemate': self.board.is_stalemate(),
            'is_game_over': self.board.is_game_over()
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_engine()


if __name__ == "__main__":
    # Exemplo de uso
    print("=== Chess Oracle - Chess Analyzer ===\n")
    
    analyzer = ChessAnalyzer()
    
    # Posição inicial
    initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print("1. Validando FEN:")
    valid, msg = analyzer.validate_fen(initial_fen)
    print(f"   {msg}\n")
    
    print("2. Informações da posição:")
    info = analyzer.get_position_info(initial_fen)
    print(f"   Turno: {info['turn']}")
    print(f"   Xeque: {info['is_check']}")
    print(f"   Lance: {info['fullmove_number']}\n")
    
    print("3. Tentando inicializar engine Maia:")
    if analyzer.initialize_engine():
        print("   Analisando posição inicial...\n")
        analysis = analyzer.analyze_position(initial_fen, depth=15)
        
        if 'error' not in analysis:
            print(f"   Avaliação: {analysis.get('evaluation', 'N/A')}")
            print(f"   Melhor lance: {analysis.get('best_move_san', 'N/A')}")
        else:
            print(f"   {analysis['error']}")
    else:
        print("   Continuando sem engine (apenas validação de posições)\n")
    
    analyzer.close_engine()
