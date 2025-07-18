import numpy as np
import torch
import ltn
import sys
from train_ltn import SudokuLTN

def load_model():
    try:
        model = SudokuLTN()
        model.load_state_dict(torch.load('sudoku_ltn.pth'))
        model.eval()
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)

model = load_model()

def is_valid_move(board, row, col, num):
    # Verifica se o número já existe na linha ou coluna
    if num in board[row, :] or num in board[:, col]:
        return False
    
    # Verifica o bloco 2x2
    block_row = 2 * (row // 2)
    block_col = 2 * (col // 2)
    block = board[block_row:block_row+2, block_col:block_col+2]
    if num in block:
        return False
    
    return True

def has_empty_cell_with_no_options(board):
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                valid_numbers = [num for num in range(1, 5) if is_valid_move(board, i, j, num)]
                if not valid_numbers:
                    return True
    return False

def classify_open(board):
    # Primeiro verifica se há células vazias sem opções válidas
    if has_empty_cell_with_no_options(board):
        return 0
    
    # Depois verifica com o modelo
    try:
        board_tensor = torch.tensor(board, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(board_tensor).item()
        return 1 if prediction > 0.5 else 0
    except Exception as e:
        print(f"Erro durante a classificação: {e}")
        return 0

if __name__ == "__main__":
    try:
        filename = sys.argv[1] if len(sys.argv) > 1 else "sudoku_open.csv"
        board = np.loadtxt(filename, delimiter=",", dtype=int)
        if board.shape != (4, 4):
            raise ValueError("O tabuleiro deve ser 4x4")
            
        result = classify_open(board)
        print("Solução possível!" if result else "Sem solução")
    except Exception as e:
        print(f"Erro: {e}")
        sys.exit(1)