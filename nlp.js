import { pipeline } from '@xenova/transformers';
import fs from 'fs/promises';

//env.localModelPath = 'models';
//env.allowRemoteModels = false;

// Pregunta
const pregunta = 'que hizo simon bolivar en paris?';

// Importamos el modelo
const QnA = await pipeline('question-answering', 'Xenova/distilbert-base-uncased-distilled-squad');
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');


// Importamos el arcvhivo de la data
const nombreArchivo = 'simon.txt';

// Leemos el contenido del archivo
const data = await fs.readFile(nombreArchivo, 'utf8');

// Obtenemos el resultado de la pregunta y su score
const { answer, score } = await QnA(pregunta, data);


// Obtenemos el embedding de la respuesta y de la pregunta
const emmbedding = await extractor(answer.split(' y ') , { pooling: 'mean', normalize: true });
const shearch = await extractor(pregunta, { pooling: 'mean', normalize: true });


// Funcion para calcular la similitud coseno
function cosinesim(A, B) {
    var dotproduct = 0;
    var mA = 0;
    var mB = 0;

    for(var i = 0; i < A.length; i++) {
        dotproduct += A[i] * B[i];
        mA += A[i] * A[i];
        mB += B[i] * B[i];
    }

    mA = Math.sqrt(mA);
    mB = Math.sqrt(mB);
    var similarity = dotproduct / (mA * mB);

    return similarity;
}

let result = []

// Calculamos la similitud coseno de cada respuesta
emmbedding.tolist().forEach((element) => {
    result.push(cosinesim(element,shearch.tolist()[0]))
});

// pintamos la respuesta, el score y la similitud coseno
console.log(answer, score, result);

