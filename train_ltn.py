import torch
import ltn
import numpy as np
from sklearn.model_selection import train_test_split

# Configuração de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================
# MODELO LTN (versão corrigida)
# ======================
class SudokuLTN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Camadas para análise de regras
        self.row_layer = torch.nn.Linear(4, 1)
        self.col_layer = torch.nn.Linear(4, 1)
        self.block_layer = torch.nn.Linear(4, 1)

        # Predicados LTN
        self.row_pred = ltn.Predicate(
            ltn.LambdaModel(lambda x: torch.sigmoid(self.row_layer(x))))
        self.col_pred = ltn.Predicate(
            ltn.LambdaModel(lambda x: torch.sigmoid(self.col_layer(x))))
        self.block_pred = ltn.Predicate(
            ltn.LambdaModel(lambda x: torch.sigmoid(self.block_layer(x))))

    def forward(self, board_tensor):
        # Garante que o tensor está no dispositivo correto
        board_tensor = board_tensor.to(device)
        result = torch.ones(1, device=device)

        # Verificação de linhas
        for i in range(4):
            row = board_tensor[i, :].unsqueeze(0)  # Shape [1,4]
            result *= self.row_pred(ltn.Variable(f"row_{i}", row)).value

        # Verificação de colunas
        for j in range(4):
            col = board_tensor[:, j].unsqueeze(0)  # Shape [1,4]
            result *= self.col_pred(ltn.Variable(f"col_{j}", col)).value

        # Verificação de blocos 2x2
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                block = board_tensor[i:i+2, j:j+2].flatten().unsqueeze(0)  # Shape [1,4]
                result *= self.block_pred(ltn.Variable(f"block_{i}{j}", block)).value

        return result

# ======================
# GERADOR DE DADOS (mantido)
# ======================
def generate_valid_board():
    base = np.array([[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]], dtype=np.float32)
    if np.random.rand() > 0.5:
        return np.rot90(base).copy()
    return base.copy()

def generate_dataset(num_samples=300):
    data = []
    for _ in range(num_samples):
        # Tabuleiros válidos (40%)
        valid = generate_valid_board()
        data.append((valid.copy(), 1.0))
        
        # Tabuleiros inválidos (40%)
        invalid = valid.copy()
        row = np.random.randint(4)
        invalid[row] = invalid[(row+1)%4]
        data.append((invalid.copy(), 0.0))
        
        # Tabuleiros abertos (20%)
        open_board = valid.copy()
        mask = np.random.choice([True, False], size=(4,4), p=[0.4, 0.6])
        open_board[mask] = 0
        data.append((open_board.copy(), 1.0))
    
    return data

# ======================
# TREINAMENTO (corrigido)
# ======================
def train():
    # 1. Preparação
    model = SudokuLTN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 2. Dataset
    dataset = generate_dataset()
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # 3. Treinamento
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    
    print(f"\nIniciando treinamento em {device}...")
    for epoch in range(100):
        model.train()
        train_loss = 0
        
        for board, label in train_data:
            optimizer.zero_grad()
            board_tensor = torch.tensor(board, dtype=torch.float32).to(device)
            target = torch.tensor([label], dtype=torch.float32).to(device)
            
            output = model(board_tensor)
            loss = torch.nn.functional.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for board, label in val_data:
                board_tensor = torch.tensor(board, dtype=torch.float32).to(device)
                target = torch.tensor([label], dtype=torch.float32).to(device)
                output = model(board_tensor)
                val_loss += torch.nn.functional.binary_cross_entropy(output, target).item()
        
        # Cálculo da média
        train_loss /= len(train_data)
        val_loss /= len(val_data)
        
        # Log
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "sudoku_ltn.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping na época {epoch+1}")
                break
    
    print("\nTreinamento concluído!")
    print(f"Melhor val_loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()