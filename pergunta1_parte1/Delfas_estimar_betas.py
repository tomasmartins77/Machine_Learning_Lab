import numpy as np
from sklearn.linear_model import LinearRegression

file1 = np.load("C:/Users/henri/OneDrive/Desktop/AAut/problem1/X_test.npy")


X_train = np.load("C:/Users/henri/OneDrive/Desktop/AAut/problem1/X_train.npy")


y_train = np.load("C:/Users/henri/OneDrive/Desktop/AAut/problem1/y_train.npy")

# Verificar a forma dos dados
print("Forma de X_train:", X_train.shape)  # Esperado: (200, 5)
print("Forma de y_train:", y_train.shape)  # Esperado: (200,)


# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Obter os coeficientes (betas) e o intercepto
betas = model.coef_           # Coeficientes para cada variável independente
intercepto = model.intercept_ # Intercepto do modelo

# Exibir os coeficientes e o intercepto
print(f"Coeficientes (Betas): {betas}")
print(f"Intercepto (Beta0): {intercepto}")