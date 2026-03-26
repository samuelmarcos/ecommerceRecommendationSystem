import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = {}
const weights = {
    price: 0.2,
    age: 0.1,
    category: 0.4,
    color: 0.3,
};

/**
 * Normaliza um valor entre 0 e 1 com base em um intervalo mínimo e máximo.
 *
 * @param {number} value - Valor a ser normalizado.
 * @param {number} min - Valor mínimo do intervalo.
 * @param {number} max - Valor máximo do intervalo.
 * @returns {number} Valor normalizado entre 0 e 1.
 */
const normalize = (value, min, max) => {
    return (value - min) / (max - min) || 1;
}

/**
 * Constrói o contexto de normalização e indexação necessário para o treinamento do modelo.
 *
 * - Calcula os valores mínimo e máximo de idade dos usuários e de preço dos produtos,
 *   usados para normalizar os dados de entrada da rede neural.
 * - Gera dicionários de índice para cores e categorias (ex.: { "preto": 0, "azul": 1 }),
 *   convertendo valores categóricos em numéricos que o modelo consegue processar.
 * - Computa a média de idade dos compradores por produto: para cada produto do catálogo,
 *   calcula a média de idade de todos os usuários que já o compraram. Quando nenhum usuário
 *   comprou o produto ainda, usa a idade média geral como valor padrão.
 *
 * @param {Array} products - Lista de produtos do catálogo (id, name, category, price, color).
 * @param {Array} users   - Lista de usuários (age, purchases[]).
 * @returns {{ minAge, maxAge, minValue, maxValue, colorIndex, categoriesIndex, productAvgAge }}
 */
function makeContext(products, users) {
    const ages = users.map(user => user.age);
    const prices = products.map(user => user.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(product => product.color))];
    const categories = [...new Set(products.map(product => product.category))];

    const colorIndex = Object.fromEntries(colors.map((color, index) => {
        return [color, index]
    }))

    const categoriesIndex = Object.fromEntries(categories.map((category, index) => {
        return [category, index]
    }))

    // Calcula a média de idade dos compradores por produto para personalizar recomendações.
    //
    // ageSums  — acumula a soma das idades de todos os usuários que compraram cada produto.
    //            { nome_do_produto: soma_das_idades }
    //
    // ageCounts — conta quantas vezes cada produto foi comprado (número de compradores).
    //             { nome_do_produto: quantidade_de_compradores }
    //
    // midAge é o ponto médio do intervalo de idades e serve como fallback para produtos
    // que ainda não foram comprados por ninguém (cold-start), evitando enviesar o modelo.

    const midAge = (maxAge - minAge) / 2;
    const ageSums = {}
    const ageCounts = {}
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    })

    // productAvgAge — dicionário { nome_do_produto: média_normalizada (0–1) }.
    // Representa o perfil etário típico de cada produto:
    //   0 → compradores jovens (próximo de minAge)
    //   1 → compradores mais velhos (próximo de maxAge)
    // Usado como feature no modelo para aprender padrões de consumo por faixa etária.
    const productAvgAge = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name]
                : midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products, users, colorIndex,
        categoriesIndex, minAge, maxAge, minPrice,
        maxPrice, numCategories: categories.length,
        //price(1) + categories(one-hot) + colors(one-hot)
        numColors: colors.length, dimensions: 1 + categories.length + colors.length
    }

}
/**
 * Cria um vetor one-hot ponderado por um peso.
 *
 * Gera um vetor de zeros com tamanho `length` onde a posição `index` recebe o valor 1,
 * depois multiplica todos os elementos pelo `weight`. Isso permite que features categóricas
 * (categoria, cor) contribuam para o vetor final com importâncias distintas.
 *
 * Exemplo: oneHotWeighted(2, 4, 0.4) → [0, 0, 0.4, 0]
 *
 * @param {number} index  - Índice da categoria ativa no vetor.
 * @param {number} length - Número total de categorias (tamanho do vetor).
 * @param {number} weight - Peso a multiplicar o vetor resultante.
 * @returns {tf.Tensor1D} Tensor 1D com o one-hot ponderado.
 */
const oneHotWeighted = (index, length, weight) => {
    return tf.oneHot(index, length).cast('float32').mul(weight);
}

/**
 * Converte um produto em um vetor numérico 1D para uso na rede neural.
 *
 * O vetor é composto por três partes concatenadas:
 *   [preço normalizado × peso] + [one-hot categoria × peso] + [one-hot cor × peso]
 *
 * Dimensão total: 1 + numCategories + numColors  (ex.: 1 + 4 + 8 = 13)
 *
 * @param {Object} product - Produto do catálogo (name, price, category, color).
 * @param {Object} context - Contexto gerado por makeContext() com índices e limites.
 * @returns {tf.Tensor1D} Tensor 1D representando o produto.
 */
function encodedProduct(product, context) {
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice) * weights.price]);

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        weights.category
    );

    const color = oneHotWeighted(
        context.colorIndex[product.color],
        context.numColors,
        weights.color
    );
    
    return tf.concat1d([price, category, color]);
}

/**
 * Converte um usuário em um vetor numérico 1D com a mesma dimensão de encodedProduct.
 *
 * Dois comportamentos dependendo do histórico de compras:
 *   - Com compras: empilha os vetores de cada produto comprado e calcula a média (mean pooling),
 *     representando o perfil de gosto do usuário. Resultado: shape [1, dimensions].
 *   - Sem compras (cold-start): cria um vetor com a idade normalizada e ponderada na posição
 *     do scalar, e zeros para categorias e cores — sinaliza ausência de preferência expressa.
 *
 * @param {Object} user    - Usuário (age, purchases[]).
 * @param {Object} context - Contexto gerado por makeContext().
 * @returns {tf.Tensor2D} Tensor de shape [1, dimensions].
 */
function encodedUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(user.purchases.map(product => 
            encodedProduct(product, context)))
            .mean(0)
            .reshape([1, context.dimensions]);
    }
    return tf.concat1d(
        [
            tf.tensor1d([normalize(user.age, context.minAge, context.maxAge) * weights.age]),
            tf.zeros([context.numCategories]),
            tf.zeros([context.numColors])
        ]
    ).reshape([1, context.dimensions]);
}

/**
 * Define a arquitetura da rede neural, compila e executa o treinamento.
 *
 * Arquitetura sequencial com 4 camadas densas:
 *   128 (relu) → 64 (relu) → 32 (relu) → 1 (sigmoid)
 *
 * A entrada tem tamanho `inputDimensions` (userVector + productVector concatenados).
 * A saída é um escalar entre 0 e 1 indicando a probabilidade de o usuário comprar o produto.
 * Treinado com otimizador Adam, loss binaryCrossentropy, por 100 épocas.
 * A cada época envia um evento `trainingLog` para a UI com loss e accuracy.
 *
 * @param {{ xs: tf.Tensor2D, ys: tf.Tensor2D, inputDimensions: number }} trainData
 * @returns {Promise<tf.Sequential>} Modelo treinado.
 */
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 128,
        inputShape: [trainData.inputDimensions],
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 64,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        verbose: 1,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: 1,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;

}

/**
 * Monta os tensores de entrada (xs) e rótulos (ys) para treinar o modelo.
 *
 * Para cada usuário que possui compras, combina o vetor do usuário com o vetor de cada produto
 * do catálogo, formando um par [userVector, productVector] como linha de entrada.
 * O rótulo (label) é 1 se o usuário comprou aquele produto, 0 caso contrário.
 *
 * Essa abordagem transforma o problema em classificação binária:
 * dado um par (usuário, produto), qual a probabilidade de compra?
 *
 * Retorna:
 *   - xs: tensor2d de shape [numPares, dimensions * 2]
 *   - ys: tensor2d de shape [numPares, 1]
 *   - inputDimensions: tamanho de cada linha de entrada (userVector + productVector)
 *
 * @param {Object} context - Contexto com users, products, vetores e dimensões.
 * @returns {{ xs: tf.Tensor2D, ys: tf.Tensor2D, inputDimensions: number }}
 */
function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
    .filter(user => user.purchases.length)
    .forEach(user => {
        const userVector = encodedUser(user, context);
        context.products.forEach(product => {
            const productVector = encodedProduct(product, context).dataSync();
            const label = user.purchases.some(purchase => purchase.name === product.name) ? 1 : 0;

            //combina usurário + produto para formar o input
            inputs.push([...userVector.dataSync(), ...productVector]);
            labels.push(label);
        })
    })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimensions: context.dimensions * 2
        //tamanho = userVector + productVector
    }
}
/**
 * Orquestra o fluxo de treinamento do modelo de recomendação.
 *
 * 1. Emite um evento de progresso (50%) para atualizar a UI.
 * 2. Busca o catálogo de produtos via fetch.
 * 3. Chama makeContext() para montar os dados normalizados.
 * 4. Emite um evento de log de época (epoch/loss/accuracy) para o painel de monitoramento.
 * 5. Após 1 segundo, emite progresso 100% e o evento de conclusão do treinamento.
 *
 * @param {{ users: Array }} param - Objeto contendo a lista de usuários para treinar o modelo.
 */
async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('../../data/products.json')).json();
    const context = makeContext(products, users);
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodedProduct(product, context).dataSync(),
        }
    })

    _globalCtx = context;

    const trainData = createTrainingData(context);

    _model = await configureNeuralNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });

}
/**
 * Gera recomendações de produtos para um usuário com base no contexto treinado.
 *
 * Recebe o perfil do usuário e o contexto global (_globalCtx) produzido pelo treinamento.
 * Deve usar o modelo para pontuar cada produto do catálogo e retornar os mais relevantes,
 * enviando o resultado de volta ao thread principal via postMessage com o evento `recommend`.
 *
 * @param {Object} user - Perfil do usuário (age, purchases[], etc.).
 * @param {Object} ctx  - Contexto de normalização e indexação gerado por makeContext().
 */
function recommend(user, ctx) {
    if(!_model) return;
    const context = _globalCtx;
    const userVector = encodedUser(user, context).dataSync();
    const input = context.productVectors.map(({vector}) => {
        return [...userVector, ...vector];
    });
    const inputsTensor = tf.tensor2d(input);
    const predictions = _model.predict(inputsTensor);
    const scores = predictions.dataSync();
    const recommendations = context.productVectors.map((item, index) => {
        return{
            ...item.meta, ...item.name, score: scores[index]
        }
    })
    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score);
    sortedRecommendations.slice(0, 10);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    });

    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

/**
 * Ponto de entrada de mensagens recebidas pelo Worker.
 *
 * Toda mensagem enviada pelo thread principal chega aqui. O campo `action` é usado
 * como chave para despachar a chamada ao handler correspondente em `handlers`.
 * Os demais campos da mensagem são passados como dados para o handler.
 */
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
