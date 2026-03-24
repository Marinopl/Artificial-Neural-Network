# 🧠 Adaline (Adaptive Linear Neuron) — Implementação em Python

Este repositório contém uma implementação do algoritmo **Adaline (Adaptive Linear Neuron)** desenvolvida do zero em Python, com suporte para:

* Batch Gradient Descent (BGD)
* Stochastic Gradient Descent (SGD)
* Avaliação de desempenho: acurácia, evolução do Erro de Classificação e evolução do Erro Quadrático Médio
* Visualização da Fronteira de Decisão aprendida

---

## 📌 Sobre o Adaline

### Definição

* O Adaline (Adaptive Linear Neuron) é um modelo de Rede Neural Artificial utilizado para tarefas de classificação binária. Assim como o Perceptron, ele busca encontrar uma fronteira de decisão linear capaz de separar classes de dados. Entretanto, diferentemente do Perceptron, o Adaline ajusta seus pesos com base no erro linear antes da aplicação da função de ativação, permitindo um processo de aprendizado mais estável e contínuo.

* Matematicamente, dado um vetor de entrada $X^{(k)} = [x_1^{(k)}, x_2^{(k)}, \dots, x_n^{(k)}]$ para uma amostra $k$ com $n$ atributos, o modelo calcula uma combinação linear dos atributos como $u = w^T \cdot X^{(k)} - \theta$, onde $w = [w_1, w_2, \dots, w_n]$ representa o vetor de pesos associados a cada entrada $x_i^{(k)}$, e $\theta$ é o bias do modelo.

* Para simplificar a implementação computacional, pode-se adicionar uma entrada fixa $x_0 = -1$, incorporando o bias ao vetor de pesos como $w_0 = \theta$. Dessa forma, o potencial de ativação passa a ser escrito como $u = w^T \cdot X^{(k)}$, onde agora $w = [w_0, w_1, \dots, w_n]$ e $X^{(k)} = [x_0^{(k)}, x_1^{(k)}, \dots, x_n^{(k)}]$.

* Diferentemente do Perceptron, durante o treinamento do Adaline a saída do modelo é considerada como sendo o próprio valor linear $y = u = w^T \cdot X^{(k)}$, sendo a função sinal aplicada apenas após o treinamento para fins de classificação: $g(u) = +1, u \geq 0$ ou $g(u) = -1, u < 0$

---

### Regra de Aprendizado do Adaline (Regra Delta)

* O treinamento do Adaline baseia-se na minimização do erro quadrático médio (Mean Squared Error — MSE), ajustando iterativamente os pesos de forma a reduzir a diferença entre a saída desejada e a saída linear do modelo.

* No Perceptron, a função de ativação define diretamente a classe prevista, de modo que o erro é calculado com base na saída após a aplicação de $g(u)$. Já no Adaline, o erro é calculado com base no valor contínuo de $u$ antes da ativação, o que permite aplicar o método do gradiente descendente para minimizar o erro quadrático médio.

* A função erro quadrático em relação às $p$ amostras de treinamento é definida por $E(w) = \frac{1}{2}\sum_{k=1}^p \left(d^{(k)} - u\right)^2$, com $d^{(k)}$ sendo o valor desejado e $u$ o valor predito pelo modelo dado por $u = w^T \cdot X^{(k)}$.

* Aplicando o operador gradiente nesta função erro, obtém-se a direção de maior crescimento do erro. Assim, para minimizá-lo, deve-se atualizar os pesos na direção oposta ao gradiente $\nabla E(w) = \frac{\partial E(w)}{\partial w} = -\sum_{k=1}^p \left(d^{(k)} - u\right)\cdot X^{(k)}$.

* A adaptação do vetor de pesos é então realizada segundo $\Delta w = -\eta \cdot \nabla E(w)$,resultando na regra de atualização $w^{\text{atual}} = w^{\text{anterior}} + \eta \sum_{k=1}^p \left(d^{(k)} - u\right)\cdot X^{(k)}$.

* A atualização dos pesos pode ser realizada de diferentes formas, mantendo o mesmo objetivo de minimizar a função erro:

    - **Batch Gradient Descent (BGD)**
        - Atualiza os pesos após processar **todas as amostras**
        - Produz convergência mais estável e suave
        - Pode ser computacionalmente mais custoso
        - Requer maior uso de memória
        - $\displaystyle w^{\text{atual}} = w^{\text{anterior}} + \eta \sum_{k=1}^p \left(d^{(k)} - u\right)\cdot X^{(k)}$

    - **Stochastic Gradient Descent (SGD)**
        - Atualiza os pesos **após cada amostra**
        - Convergência mais rápida, porém mais ruidosa
        - Menor custo de memória
        - Indicado para grandes volumes de dados
        - $\displaystyle w^{\text{atual}} = w^{\text{anterior}} + \eta \left(d^{(k)} - u\right)\cdot X^{(k)}, \quad k=1,\dots,p$

* Como o algoritmo busca encontrar iterativamente o vetor de pesos ótimo $w^*$, é necessário estabelecer um critério de parada baseado na convergência do erro quadrático médio definido por $E_{\text{qm}}(w) = \frac{1}{p}\sum_{k=1}^p \left(d^{(k)} - u\right)^2$.

O processo iterativo é interrompido quando a variação do erro entre duas épocas consecutivas satisfaz $\left| E_{\text{qm}}(w^{\text{atual}}) - E_{\text{qm}}(w^{\text{anterior}}) \right| \leq \varepsilon$, onde $\varepsilon$ é a precisão requerida para o processo de convergência.

---

## ⚙️ Funcionalidades

* Implementação do Adaline com:

  * Gradiente Batch (BGD)
  * Gradiente Estocástico (SGD)
* Inclusão automática do bias
* Critério de parada baseado na variação do Erro Quadrático Médio
* Cálculo de:

  * Erro quadrático médio (MSE)
  * Erro de classificação
  * Acurácia
* Plot da evolução do erro ao longo das épocas
* Visualização da fronteira de decisão (para 2 features)

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
