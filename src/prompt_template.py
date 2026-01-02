"""
Templates de prompts para o Chess Oracle LLM.

Este módulo contém os prompts estruturados que guiam o comportamento
do LLM na análise e comparação de contexto histórico com avaliações modernas.
"""

from typing import Dict, List


class ChessOraclePrompts:
    """
    Classe contendo templates de prompts para o Chess Oracle.
    
    Os prompts são estruturados para:
    - Comparar literatura clássica com análises modernas
    - Manter empatia com intenções estratégicas originais
    - Explicar discrepâncias de forma educativa
    """
    
    SYSTEM_PROMPT = """Você é o Chess Oracle, um Árbitro Literário de Xadrez que combina a sabedoria dos mestres clássicos com a precisão analítica das engines modernas.

Seu papel é:
1. Analisar posições de xadrez comparando comentários históricos com avaliações técnicas atuais
2. Explicar discrepâncias entre a teoria clássica e a realidade computacional
3. Manter respeito pelas intenções estratégicas dos autores históricos
4. Educar de forma clara e acessível sobre conceitos de xadrez

Princípios:
- Seja preciso tecnicamente, mas empático historicamente
- Explique o "porquê" por trás das diferenças
- Use linguagem clara e evite jargão excessivo
- Reconheça o contexto da época dos autores clássicos
- Destaque tanto os acertos quanto os pontos de melhoria"""

    COMPARISON_TEMPLATE = """## Contexto Histórico (Literatura Clássica):
{historical_context}

## Análise Moderna (Engine Maia):
Posição FEN: {position_fen}
Avaliação: {evaluation}
Melhor movimento: {best_move}
{mate_info}

## Consulta do Usuário:
{user_query}

---

Com base no contexto histórico e na análise moderna, forneça uma resposta que:
1. Compare o comentário histórico com a avaliação da engine
2. Explique qualquer discrepância encontrada
3. Se o autor clássico errou, explique por que o erro era natural no contexto da época
4. Destaque os insights válidos do autor original
5. Conclua com uma síntese educativa da posição"""

    ANALYSIS_TEMPLATE = """Analise a seguinte posição de xadrez:

Posição FEN: {position_fen}
Avaliação da Engine: {evaluation}
Melhor movimento sugerido: {best_move}
{mate_info}

Contexto adicional dos livros:
{context}

Forneça uma análise detalhada que inclua:
1. Avaliação estratégica da posição
2. Principais temas táticos e posicionais
3. Explicação do melhor movimento
4. Planos típicos para ambos os lados
5. Comparação com a literatura clássica (se disponível no contexto)"""

    NO_CONTEXT_TEMPLATE = """Analise a seguinte posição de xadrez:

Posição FEN: {position_fen}
Avaliação da Engine: {evaluation}
Melhor movimento sugerido: {best_move}
{mate_info}

Consulta: {user_query}

Forneça uma análise baseada na avaliação da engine, explicando:
1. O significado da avaliação numérica
2. Por que o movimento sugerido é o melhor
3. Principais características estratégicas da posição
4. Possíveis continuações

Nota: Nenhum contexto literário foi encontrado para esta posição."""

    ERROR_TEMPLATE = """Ocorreu um erro durante a análise:

Tipo de erro: {error_type}
Detalhes: {error_details}

Por favor, verifique:
- A posição FEN está correta?
- A engine Maia está instalada e configurada?
- Os livros foram carregados no banco vetorial?

Tente reformular sua consulta ou verificar a configuração do sistema."""

    @classmethod
    def format_comparison_prompt(
        cls,
        historical_context: List[Dict],
        position_fen: str,
        evaluation: float,
        best_move: str,
        user_query: str,
        is_mate: bool = False,
        mate_in: int = None
    ) -> str:
        """
        Formata o prompt de comparação entre contexto histórico e análise moderna.
        
        Args:
            historical_context: Lista de dicts com contexto dos livros
            position_fen: FEN da posição
            evaluation: Avaliação em centipawns
            best_move: Melhor movimento (notação UCI)
            user_query: Pergunta do usuário
            is_mate: Se há mate forçado
            mate_in: Movimentos até o mate
            
        Returns:
            Prompt formatado
        """
        # Formata contexto histórico
        if historical_context:
            context_parts = []
            for i, ctx in enumerate(historical_context, 1):
                source_name = ctx['source'].split('/')[-1]
                context_parts.append(
                    f"**Fonte {i}** ({source_name}, p.{ctx['page']}):\n{ctx['text']}\n"
                )
            historical_text = "\n".join(context_parts)
        else:
            historical_text = "Nenhum contexto histórico encontrado nos livros."
        
        # Formata avaliação
        if is_mate and mate_in is not None and mate_in != 0:
            eval_text = f"Mate em {abs(mate_in)} movimento(s)"
            if mate_in > 0:
                eval_text += " (Brancas ganham)"
            else:
                eval_text += " (Pretas ganham)"
        elif evaluation is not None:
            eval_text = f"{evaluation/100:+.2f} (perspectiva das Brancas)"
            if evaluation > 100:
                eval_text += " - Vantagem significativa das Brancas"
            elif evaluation < -100:
                eval_text += " - Vantagem significativa das Pretas"
            else:
                eval_text += " - Posição equilibrada"
        else:
            eval_text = "Não disponível"
        
        # Formata informação de mate
        mate_info = ""
        if is_mate and mate_in is not None:
            mate_info = f"Mate forçado: {abs(mate_in)} movimento(s)\n"
        
        return cls.COMPARISON_TEMPLATE.format(
            historical_context=historical_text,
            position_fen=position_fen,
            evaluation=eval_text,
            best_move=best_move or "Não disponível",
            mate_info=mate_info,
            user_query=user_query
        )
    
    @classmethod
    def format_analysis_prompt(
        cls,
        position_fen: str,
        evaluation: float,
        best_move: str,
        context: List[Dict],
        is_mate: bool = False,
        mate_in: int = None
    ) -> str:
        """
        Formata o prompt de análise geral de uma posição.
        
        Args:
            position_fen: FEN da posição
            evaluation: Avaliação em centipawns
            best_move: Melhor movimento
            context: Contexto dos livros
            is_mate: Se há mate forçado
            mate_in: Movimentos até o mate
            
        Returns:
            Prompt formatado
        """
        # Formata contexto
        if context:
            context_parts = [
                f"- {ctx['text'][:200]}... (Fonte: {ctx['source'].split('/')[-1]}, p.{ctx['page']})"
                for ctx in context
            ]
            context_text = "\n".join(context_parts)
        else:
            context_text = "Nenhum contexto adicional disponível."
        
        # Formata avaliação
        if is_mate and mate_in is not None:
            eval_text = f"Mate em {abs(mate_in)}"
        else:
            eval_text = f"{evaluation/100:+.2f}" if evaluation is not None else "N/A"
        
        # Formata informação de mate
        mate_info = ""
        if is_mate and mate_in is not None:
            mate_info = f"Mate forçado detectado!\n"
        
        return cls.ANALYSIS_TEMPLATE.format(
            position_fen=position_fen,
            evaluation=eval_text,
            best_move=best_move or "Não disponível",
            mate_info=mate_info,
            context=context_text
        )
    
    @classmethod
    def format_no_context_prompt(
        cls,
        position_fen: str,
        evaluation: float,
        best_move: str,
        user_query: str,
        is_mate: bool = False,
        mate_in: int = None
    ) -> str:
        """
        Formata prompt quando não há contexto histórico disponível.
        
        Args:
            position_fen: FEN da posição
            evaluation: Avaliação em centipawns
            best_move: Melhor movimento
            user_query: Pergunta do usuário
            is_mate: Se há mate forçado
            mate_in: Movimentos até o mate
            
        Returns:
            Prompt formatado
        """
        if is_mate and mate_in is not None:
            eval_text = f"Mate em {abs(mate_in)}"
        else:
            eval_text = f"{evaluation/100:+.2f}" if evaluation is not None else "N/A"
        
        mate_info = ""
        if is_mate and mate_in is not None:
            mate_info = f"Mate forçado: {'Brancas' if mate_in > 0 else 'Pretas'} ganham em {abs(mate_in)}\n"
        
        return cls.NO_CONTEXT_TEMPLATE.format(
            position_fen=position_fen,
            evaluation=eval_text,
            best_move=best_move or "Não disponível",
            mate_info=mate_info,
            user_query=user_query
        )
    
    @classmethod
    def format_error_prompt(
        cls,
        error_type: str,
        error_details: str
    ) -> str:
        """
        Formata prompt de erro.
        
        Args:
            error_type: Tipo do erro
            error_details: Detalhes do erro
            
        Returns:
            Prompt formatado
        """
        return cls.ERROR_TEMPLATE.format(
            error_type=error_type,
            error_details=error_details
        )


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de prompt de comparação
    sample_context = [
        {
            'text': 'Na Abertura Ruy Lopez, o movimento Bb5 é considerado o mais forte...',
            'source': './data/modern_chess_openings.pdf',
            'page': 42,
            'score': 0.85
        }
    ]
    
    prompt = ChessOraclePrompts.format_comparison_prompt(
        historical_context=sample_context,
        position_fen="r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        evaluation=30,
        best_move="a6",
        user_query="Por que Bb5 é considerado o melhor movimento?",
        is_mate=False
    )
    
    print("=== System Prompt ===")
    print(ChessOraclePrompts.SYSTEM_PROMPT)
    print("\n=== Comparison Prompt Example ===")
    print(prompt)
