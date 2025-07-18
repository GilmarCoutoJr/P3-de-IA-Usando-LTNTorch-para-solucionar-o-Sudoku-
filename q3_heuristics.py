import numpy as np
import torch
from train_ltn import SudokuLTN

def is_valid_move(board, row, col, num):
    if num in board[row] or num in board[:,col]:
        return False
    block_row, block_col = 2*(row//2), 2*(col//2)
    return num not in board[block_row:block_row+2, block_col:block_col+2]

def evaluate_move(model, board, row, col, num):
    temp = board.copy()
    temp[row,col] = num
    with torch.no_grad():
        score = model(torch.tensor(temp, dtype=torch.float32)).item()
    
    # Fatores heurísticos
    factors = 0
    if len(np.unique(board[row])) == 3: factors += 0.1
    if len(np.unique(board[:,col])) == 3: factors += 0.1
    block = board[2*(row//2):2*(row//2)+2, 2*(col//2):2*(col//2)+2]
    if len(np.unique(block)) == 3: factors += 0.1
    
    return min(0.99, score + factors)

def recommend_move(board, model):
    best_cell = None
    best_num = None
    best_score = -1
    
    for i in range(4):
        for j in range(4):
            if board[i,j] == 0:
                for num in range(1,5):
                    if is_valid_move(board, i, j, num):
                        score = evaluate_move(model, board, i, j, num)
                        if score > best_score:
                            best_score = score
                            best_cell = (i,j)
                            best_num = num
    
    if best_cell:
        print(f"Melhor jogada: Célula {best_cell}")
        print(f"Número recomendado: {best_num}")
        print(f"Confiança: {best_score*100:.1f}%")
        return best_cell, best_num
    return None

if __name__ == "__main__":
    model = SudokuLTN()
    model.load_state_dict(torch.load('sudoku_ltn.pth'))
    model.eval()
    
    test_board = np.array([
        [1, 0, 0, 2],
        [0, 0, 0, 4],
        [4, 0, 2, 0],
        [0, 0, 0, 1]
    ])
    
    recommend_move(test_board, model)