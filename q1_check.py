import numpy as np
import torch
import ltn
import sys
from train_ltn import SudokuLTN

def load_model():
    model = SudokuLTN()
    model.load_state_dict(torch.load('sudoku_ltn.pth'))
    model.eval()
    return model

def is_valid_sudoku(board, model):
    board_tensor = torch.tensor(board, dtype=torch.float32)
    
    # Verificar todas as linhas
    rows_valid = all(
        model.row_pred(ltn.Variable("x", board_tensor[i,:].unsqueeze(0))).value.item() > 0.5
        for i in range(4)
    )
    
    # Verificar todas as colunas
    cols_valid = all(
        model.col_pred(ltn.Variable("x", board_tensor[:,j].unsqueeze(0))).value.item() > 0.5
        for j in range(4)
    )
    
    # Verificar todos os blocos
    blocks_valid = all(
        model.block_pred(ltn.Variable("x", board_tensor[i:i+2,j:j+2].flatten().unsqueeze(0))).value.item() > 0.5
        for i in range(0,4,2) for j in range(0,4,2)
    )
    
    return 1 if (rows_valid and cols_valid and blocks_valid) else 0

if __name__ == "__main__":
    # Permite usar argumento do terminal
    filename = sys.argv[1] if len(sys.argv) > 1 else "sudoku_valid.csv"
    board = np.loadtxt(filename, delimiter=",", dtype=int)
    model = load_model()
    print("Tabuleiro válido!" if is_valid_sudoku(board, model) else "Inválido")
