mod layer;
use layer::Layer;

use rand::Rng;

fn test() {
    let inputs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let output = [vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let mut input_layer: Layer = Layer::new(&inputs[0].to_vec(), 3);
    input_layer.forward();
    
    let mut hidden_layer: Layer = Layer::new(&input_layer.outputs, 1);
    hidden_layer.forward();

    let mut r = 0;

    for i in 0..1000 {
        hidden_layer.backward(&output[r]);
        input_layer.backward(&hidden_layer.outputs);
        hidden_layer.outputs.clear();
        input_layer.outputs.clear();

        r = rand::thread_rng().gen_range(0..3);

        input_layer.inputs = inputs[r].to_vec();
        
        input_layer.forward();
        hidden_layer.forward();
        println!("out: {:?}", hidden_layer.outputs);
    }
    
}

fn main() {
    test();
}