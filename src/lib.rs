use std::cell::RefCell;
use std::rc::Rc;

mod op;
pub use op::Op;

/// Output of a node operation, called tensor
///
/// This points to a `Node` struct backed by a graph
pub struct Tensor {
    graph: Rc<RefCell<Graph>>,
    id: usize
}

impl Tensor {
    pub fn shape(&self) -> Vec<usize> {
        self.graph.borrow().node_set[self.id].shape.clone()
    }
}

/// Manages buffers and structure of a node inside the graph
pub struct Node {
    // unique identification in the graph
    id: usize,
    // performing operation
    op: Box<dyn Op>,
    // shape of the output tensor
    shape: Vec<usize>,
    // back-pointer to output of previous tensor
    input_nodes: Vec<Tensor>,
}

pub struct Graph {
    node_set: Vec<Node>
}

impl Graph {
    pub(crate) fn add(&mut self, mut node: Node) -> usize {
        let id = self.node_set.len();
        node.id = id;
        self.node_set.push(node);
        id
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        // create a new graph structure living on the CPU
        let graph = en::Graph::on_cpu();

        // load the input and trainable parameter to the CPU
        let input = graph.load(Array2::zeros((5, 5))).await;
        let parameter = graph.load_trainable(Array2::zeros((5, 5))).await;

        let rand = graph.rand_uniform((5, 5));

        // define network
        let output = input.mm(&parameter);

        // calculate output
        let result = output.await;

        // calculate gradient for every node
        output.backward();

        // update trainable parameters and zero gradient
        for (tensor, update_tensor) in output.iter_trainable() {
            tensor.weights += 0.01 * update_tensor;
            update_tensor.zero();
        }
    }
}
