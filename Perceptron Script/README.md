# 🧠 Perceptron — Implementação em Python

Este repositório contém uma implementação do algoritmo **Perceptron** desenvolvida do zero em Python, com suporte para:

* Avaliação de desempenho: acurácia, evolução do Erro de Classificação
* Visualização da Fronteira de Decisão aprendida

---

## 📌 Sobre o Adaline

### Definição

* O Perceptron é um dos modelos mais simples de Rede Neural Artificial, utilizado para tarefas de classificação binária. Ele busca encontrar uma fronteira de decisão linear (classificador linear) que separe duas ou mais classes de dados. Lembrando que, para dois atributos, o separador é uma reta, para três, um plano, e para dimensões maiores, temos um hiperplano.

* Matematicamente, dado um vetor de entrada $X^{(k)} = [x_1^{(k)}, x_2^{(k)}, \dots, x_n^{(k)}]$ para uma amostra $k$ com $n$ atributos,
o modelo calcula uma combinação linear dos atributos como  $u = w^T \cdot X^{(k)} - \theta,$ onde $w = [w_1, w_2, \dots, w_n]$ representa o vetor de pesos associados a cada entrada $x_i$, e $\theta$ é o bias associado ao modelo.

* De forma geral, para facilitar o algoritmo e evitar um loop extra,
podemos implementar ao vetor de entrada uma entrada adicional fixa $x_0 = -1$, de forma que o bias possa ser adicionado ao vetor de pesos como $w_0 = \theta$. Assim, o potencial de ativação toma a seguinte forma $u = w^T \cdot X^{(k)}$, onde $w = [w_0, w_1, \dots, w_n]$ e $X^{(k)} = [x_0^{(k)}, x_1^{(k)}, \dots, x_n^{(k)}]$.

* Por fim, a saída do modelo é obtida por meio da função de ativação sinal: $y = g(u) = +1, u \geq 0$ ou $ y = g(u) = -1, u < 0$

### Regra de Aprendizado de Hebb

* O treinamento do Perceptron consiste em ajustar iterativamente os pesos com base nos erros de classificação.
* Para cada amostra $(X^{(k)}, d^{(k)})$, os pesos são atualizados segundo a seguinte regra: $w \leftarrow w + \eta \cdot (d^{(k)} - y) \cdot X^{(k)}$
onde $\eta$ é a taxa de aprendizado, $d^{(k)}$ é o valor desejado e $y$ é a saída predita pelo modelo. A atualização dos pesos ocorre apenas quando o modelo comete um erro, isto é, quando $d^{(k)}\neq y$.

### Acurácia e Erro de Classificação

* A acurácia do modelo calcula-se como acc = predições corretas / número de amostras

* Para acompanhar a evolução do erro durante o treinamento, define-se o erro de classificação como erro = número de amostras classificadas incorretamente / total de amostras

---

## 📂 Estrutura dos Dados

Os dados devem estar organizados da seguinte forma:

* **X**: matriz com dimensão `(n_features, n_samples)`
* **y**: vetor linha `(1, n_samples)`

sendo **X** o vetor de entrada e **y** os labels esperados.

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 👨‍💻 Autor

Desenvolvido como estudo prático de Redes Neurais Artificiais.

---
