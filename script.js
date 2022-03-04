function hexToRgb(hex) {
    const normal = hex.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (normal) return normal.slice(1).map(e => parseInt(e, 16));
  
    const shorthand = hex.match(/^#([0-9a-f])([0-9a-f])([0-9a-f])$/i);
    if (shorthand) return shorthand.slice(1).map(e => 0x11 * parseInt(e, 16));
  
    return null;
}

const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [3],
    activation: "relu",
    units: 25
}));
model.add(tf.layers.dense({
    inputShape: [25],
    activation: "relu",
    units: 25
}));
model.add(tf.layers.dense({
    inputShape: [25],
    activation: "sigmoid",
    units: 1
}));
model.compile({
    optimizer: tf.train.adam(),
    loss: "meanSquaredError"
});

(async() => {
    const data = [
        { r: 255, g: 255, b: 255, light: 0 },
        {r: 255, g: 0, b: 0, light: 0},
        {r: 0, g: 0, b: 0, light: 1},
        {r: 42, g: 254, b: 45, light: 0},
        {r: 107, g: 230, b: 255, light: 0},
        {r: 38, g: 28, b: 79, light: 1},
        {r: 66, g: 21, b: 56, light: 1},
        {r: 96, g: 1, b: 1, light: 1},
        {r: 29, g: 27, b: 27, light: 1},
        {r: 52, g: 50, b: 50, light: 1},
        {r: 131, g: 47, b: 116, light: 1},
        {r: 223, g: 1, b: 179, light: 0}
    ];

    const x = [];
    const y = [];
    
    for (const point of data) {
        x.push([point.r / 255, point.g / 255, point.b / 255]);
        y.push(point.light);    
    }

    console.log("Data X:", x);
    console.log("Data Y:", y);

    console.log(await model.fit(tf.tensor2d(x), tf.tensor1d(y), {
        epochs: 250
    }));

    console.log("Trained!");

    document.getElementById("picker").disabled = false;
})();

document.getElementById("picker").addEventListener("change", async event => {
    const [r, g, b] = hexToRgb(event.target.value);
    console.log("Colour:", { r, g, b });
    document.getElementById("target").style.backgroundColor = event.target.value;

    const predicted = (await model.predict(tf.tensor2d([[r / 255, g / 255, b / 255]]))).dataSync()[0];
    console.log("Network predicted:", predicted);

    document.getElementById("text").style.color = Math.round(predicted) > 0.5 ? 'white' : 'black';
});