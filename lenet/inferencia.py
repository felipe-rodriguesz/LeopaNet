import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import random

# 1. Devemos recriar a arquitetura base da rede. O PyTorch precisa da "casca"
#    com os mesmos tamanhos exatos antes de carregar e injetar os "pesos" salvos.
class LeopaNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(LeopaNet, self).__init__()
        # Mesmas configurações do notebook
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Detecta automaticamente se há placa de vídeo ou se carregará na CPU comum
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando em modo de inferência com: {device}")

    # 2. Instanciamos a casca do modelo
    model = LeopaNet().to(device)

    # 3. Carregamos o "cérebro" (state_dict com os pesos e bias aprendidos)
    caminho_modelo = "leopanet_mnist.pth"
    try:
        # map_location garante que seja convertido perfeitamente caso tenha treinado na GPU e rode na CPU no futuro
        model.load_state_dict(torch.load(caminho_modelo, map_location=device, weights_only=True))
        print(f"✅ O modelo '{caminho_modelo}' foi injetado na memória com sucesso!")
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{caminho_modelo}' não foi localizado no diretório atual.")
        return

    # 4. Modo de avaliação da rede! 
    # (congela o autograd de aprendizado e ajusta camadas como Dropout, se existissem)
    model.eval()

    # -- VAMOS FAZER O TESTE NA PRÁTICA! --
    # Carregando uma amostra do MNIST para teste de fogo
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # Sorteia uma imagem qualquer das 10.000 existentes no Teste
    idx_aleatorio = random.randint(0, len(test_dataset) - 1)
    imagem, rotulo_real = test_dataset[idx_aleatorio]
    
    # A LeopaNet espera receber (Batch, Canais, H, W). 
    # A imagem solta é (1, 28, 28). O unsqueeze vai deixá-la (1, 1, 28, 28)
    imagem_batch = imagem.unsqueeze(0).to(device)

    # Executa a rede usando inferência blindada sem gerar autografos de gradiente
    with torch.no_grad():
        saida_logits = model(imagem_batch)
        
        # Pega qual dos 10 canais finais de saída teve o valor mais forte
        valor_forca, previu = saida_logits.max(1)
        
    print(f"\n--- 🎯 RESULTADO DO TESTE RÁPIDO ---")
    print(f"Qual é o Rótulo Verdadeiro?: {rotulo_real}")
    print(f"O que a sua LeopaNet 'viu'?: {previu.item()}")
    
    # Plotar o digito manuscrito sorteado
    img_plot = imagem.squeeze().numpy() * 0.3081 + 0.1307 # Restaurar paleta original
    plt.imshow(img_plot, cmap='gray')
    cor = 'green' if rotulo_real == previu.item() else 'red'
    plt.title(f"Anotação: {rotulo_real} | LeopaNet leu como: {previu.item()}", color=cor, fontsize=14)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
