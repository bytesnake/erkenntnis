use std::cell::RefCell;
use std::rc::Rc;

mod op;
pub use op::Op;

pub struct Tensor {
    graph: Rc<RefCell<Graph>>,
    id: usize
}

impl Tensor {
    pub fn shape(&self) -> Vec<usize> {
        self.graph.borrow().node_set[self.id].shape.clone()
    }
}

pub struct TensorInternal {
    id: usize,
    op: Box<dyn Op>,
    shape: Vec<usize>,
    in_indices: Vec<usize>,

}

pub struct Graph {
    node_set: Vec<TensorInternal>
}

impl Graph {
    pub(crate) fn add(&mut self, mut node: TensorInternal) -> usize {
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
        assert_eq!(2 + 2, 4);
    }
}
