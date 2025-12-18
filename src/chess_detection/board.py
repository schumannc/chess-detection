from typing import Any
import chess
import chess.svg
import gradio as gr


def render_board(board: chess.Board | None = None) -> str:
    if board is None:
        board = chess.Board()
    svg = chess.svg.board(board=board, size=420)
    return svg


def update_board(positions: list[bool], board: chess.Board) -> tuple[Any, chess.Board]:
    detected_occupied = set()
    for i, has_piece in enumerate(positions):
        if has_piece:
            detected_occupied.add(i)
    
    previous_occupied = set()
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            previous_occupied.add(square)

    print("Previous occupied:", previous_occupied)
    print("Detected occupied:", detected_occupied)
    
    disappeared = previous_occupied - detected_occupied  # Pieces that left
    appeared = detected_occupied - previous_occupied     # Pieces that appeared
    
    if len(disappeared) == 1 and len(appeared) == 1:
        # Simple move: one piece disappeared, one appeared
        from_square = disappeared.pop()
        to_square = appeared.pop()
        
        # Create and validate the move
        move = chess.Move(from_square, to_square)
        
        # Check for pawn promotion (if pawn reaches last rank)
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or \
               (piece.color == chess.BLACK and chess.square_rank(to_square) == 0):
                # Default to queen promotion
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
        
        # Validate and make the move if legal
        if move in board.legal_moves:
            board.push(move)
        else:
            # Try as capture or special move
            for legal_move in board.legal_moves:
                if legal_move.from_square == from_square and legal_move.to_square == to_square:
                    board.push(legal_move)
                    break
    
    elif len(disappeared) == 2 and len(appeared) == 2:
        # Could be castling: king and rook both move
        disappeared_list = list(disappeared)
        appeared_list = list(appeared)
        
        # Try to find a legal castling move
        for legal_move in board.legal_moves:
            if board.is_castling(legal_move):
                # Check if this castling move matches the detected changes
                from_sq = legal_move.from_square
                to_sq = legal_move.to_square
                
                # Get the rook squares involved in castling
                if board.piece_at(from_sq) and board.piece_at(from_sq).piece_type == chess.KING:
                    if from_sq in disappeared_list and to_sq in appeared_list:
                        board.push(legal_move)
                        break
    
    elif len(disappeared) == 2 and len(appeared) == 1:
        # Likely a capture: one piece captured, attacker moved
        to_square = appeared.pop()
        
        # Try to find which piece moved
        for from_square in disappeared:
            move = chess.Move(from_square, to_square)
            
            # Check for pawn promotion with capture
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and chess.square_rank(to_square) == 7) or \
                   (piece.color == chess.BLACK and chess.square_rank(to_square) == 0):
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            
            if move in board.legal_moves:
                board.push(move)
                break
    
    # Render the updated board
    svg = render_board(board)
    return svg, board