import chess
import chess.svg
import gradio as gr


def render_board(fen:None|str=None):
    board = chess.Board()
    svg = chess.svg.board(board=board, size=420)
    return svg