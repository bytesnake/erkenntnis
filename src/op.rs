use std::any::type_name;
use std::rc::Rc;
use std::cell::RefCell;

use crate::{Graph, Tensor};

pub trait Op {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn compute(&self, ctx: &mut ComputeContext);
    //fn grad(&self, ctx: &mut GradientContext);
}

pub struct ComputeContext {
    inputs: Vec<Tensor>,
    graph: Rc<RefCell<Graph>>
}
