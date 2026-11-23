# SVM não linear e kernel RBF: como C e γ controlam a fronteira de decisão

Investigação experimental, com visualização em 2D, de como os hiperparâmetros **C** e **γ** controlam a flexibilidade da fronteira de decisão de uma SVM com kernel RBF. 

> **Pergunta central:** como C e γ controlam a flexibilidade da fronteira produzida pela SVM?

## Dados

| Conjunto | Origem | Características |
|---|---|---|
| **moons** | `make_moons` | Duas meias-luas, `noise=0.30`. Há sobreposição real entre as classes, então nenhum modelo acerta 100% em dados novos e o sobreajuste fica visível |
| **circles** | `make_circles` | Círculo dentro de círculo, `noise=0.12`, `factor=0.5` |
| **Breast Cancer Wisconsin** | `load_breast_cancer` (sklearn) | 569 exemplos e 30 atributos em escalas muito diferentes, usado só no experimento de padronização |

Cada conjunto sintético tem 1.000 pontos, com classes balanceadas. Tudo é gerado ou carregado pelo próprio notebook, então **não há arquivos de dados externos**.

## Protocolo

- **Divisão estratificada (semente 42):** 60% treino · 20% validação · 20% teste.
- **Treino** ajusta os modelos dos experimentos exploratórios, avaliados na validação.
- **Treino + validação** (800 pontos) é usado na comparação de kernels e na seleção final de C e γ, com validação cruzada estratificada de 5 partes.
- **Teste** é usado **uma única vez**, na avaliação final.
- **Padronização** (`StandardScaler`) dentro de um `Pipeline`, ajustada só com os dados de treino de cada ajuste, sem vazamento.
- **Métricas:** acurácia e F1 (classes balanceadas), além do número de vetores de suporte (VS) e do tempo de treino.

## Experimentos e resultados

### 1. Kernels: linear × polinomial × RBF

Cada kernel recebeu sua própria busca de hiperparâmetros (validação cruzada em treino + validação).

| Conjunto | Linear | Polinomial | RBF |
|---|---|---|---|
| moons | 0,836 | 0,906 | **0,909** |
| circles | 0,534 | 0,976 | **0,978** |

O kernel linear falhou nos dois conjuntos (no *circles*, ficou próximo do acaso). Polinomial e RBF empataram dentro da variação da validação cruzada. A vantagem da RBF é não exigir a escolha de um grau polinomial: a flexibilidade vem de C e γ.

### 2. C fixo (C = 1), variando γ

- **γ pequeno** → fronteira quase linear, com **subajuste** (γ = 0,01: treino 0,845 e validação 0,800).
- **γ intermediário** → melhor equilíbrio, com **0,900 de validação** em γ ≈ 18.
- **γ grande** → fronteira com "ilhas" ao redor dos pontos, com **sobreajuste** (γ = 1000: treino 0,998 e validação 0,705; 99% dos pontos viram VS).

### 3. γ fixo (γ = 1), variando C

- **C pequeno** → margem larga e muitas violações (C = 0,01: 598 dos 600 pontos viram VS).
- **C grande** → margem estreita: a largura caiu **~1.400 vezes** entre C = 10⁻³ e C = 10⁴, e as violações caíram de 600 para 88.
- A validação sobe até C ≈ 0,3 (**0,885**) e cai para 0,850 em C = 10⁴, indicando início de sobreajuste.

### 4. Interação C × γ (grade 13 × 13)

`GridSearchCV` com validação cruzada de 5 partes sobre 169 combinações (C de 10⁻² a 10⁴ e γ de 10⁻³ a 10³).

- Há uma **faixa diagonal de boas combinações** no plano log C × log γ: γ menor pode ser compensado por C maior, e vice-versa. Essa compensação tem limite, e nos extremos de γ variar C não resolve.
- Melhor combinação no *moons*: **C ≈ 0,32 e γ ≈ 3,2**, com acurácia de VC de **0,912 ± 0,026**. As 10 melhores combinações diferem menos que o desvio da validação cruzada.

### 5. Importância da padronização

| Cenário | Resultado |
|---|---|
| *moons* com x₁ × 100, sem padronizar, γ = 1 | treino 0,957 × validação 0,855 (sobreajuste) |
| *moons* com x₁ × 100, sem padronizar, `gamma="scale"` | validação 0,655 (subajuste) |
| *moons* com x₁ × 100, padronizado, γ = 1 | validação 0,880 |
| Breast Cancer, sem padronizar (C e γ otimizados) | 0,947 |
| Breast Cancer, padronizado (C e γ otimizados) | **0,979** |

Atributos de maior escala dominam as distâncias do kernel, e nem a escolha automática de γ nem a otimização compensam isso.

### 6. Tempo e complexidade

- **Tempo × C:** treino rápido (≈ 10 ms) para C pequeno, mas 25 a 45 vezes mais lento com C = 10⁴.
- **Escalabilidade:** o tempo de treino cresce aproximadamente com **n²·¹**. Dobrar os dados multiplica o tempo por ~4 a 5 (de 4,0 ms com n = 500 para 3,3 s com n = 16.000).
- **Complexidade:** o número de VS cresce proporcionalmente aos dados (~20 a 25%), e o custo de predição é proporcional ao número de VS.

### Avaliação final (conjunto de teste, usado uma vez)

| Conjunto | C* | γ* | Acurácia (VC) | **Acurácia (teste)** | F1 (teste) | VS |
|---|---|---|---|---|---|---|
| moons | 0,32 | 3,2 | 0,912 | **0,935** | 0,934 | 284 (35%) |
| circles | 31,6 | 0,032 | 0,980 | **0,985** | 0,985 | 129 (16%) |

A seleção se adapta ao problema: o γ escolhido para o *circles* é ~100× menor que o do *moons* (uma forma suave não precisa de gaussianas estreitas), e o C maior compensa, o mesmo efeito da faixa diagonal.

## Conclusões

1. **γ controla o alcance da influência dos vetores de suporte** e foi o parâmetro que mais alterou a flexibilidade da fronteira: γ pequeno leva a subajuste, γ grande a sobreajuste.
2. **C controla o compromisso entre margem larga e penalização dos erros de treino:** C pequeno dá margem larga e fronteira suave, C grande dá margem estreita e esforço maior para acomodar os exemplos.
3. **C e γ interagem** e devem ser **ajustados em conjunto**, por busca em grade logarítmica com validação cruzada.
4. **Kernels não lineares** são necessários nesses problemas, e a RBF dispensa a escolha do grau polinomial.
5. **A padronização é essencial** com kernel RBF, e deve ser feita dentro do `Pipeline` para evitar vazamento.
6. **O custo cresce rápido com n** (~n²), e o número de VS determina o custo de predição.

**Limitações** (discutidas no notebook): a validação hold-out tem só 200 pontos, e o desvio da validação cruzada (~0,03) é maior que várias das diferenças comparadas. A grade é discreta, e os tempos dependem da máquina e da implementação (libsvm). As classes são balanceadas, então problemas desbalanceados exigiriam outras métricas e `class_weight`.

## Como executar

```bash
git clone https://github.com/<seu-usuario>/<nome-do-repositorio>.git
cd <nome-do-repositorio>
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook Projeto_4_RBF_e_SVM.ipynb
```

Todas as sementes são fixas (42), então os resultados são reprodutíveis. A busca em grade completa (845 ajustes) leva cerca de 20 segundos.

## Estrutura

```
.
├── RBF_e_SVM.ipynb
└── README.md
```

## Tecnologias

Python · scikit-learn · NumPy · pandas · Matplotlib

## Autoria

- Anysabele de Paula Barbosa Santos
- Fátima Regina Queiroz de Araújo Lima
- Millena Ferreira Marçal das Neves
